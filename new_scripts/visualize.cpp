// visualize.cpp
// Build an interactive plot or a video (GIF/MP4) of the system using gnuplot.
//
// Usage:
//   ./visualize --model=FILE [options...]
//
// Only --model is required. All other options have sensible defaults that
// produce a 300-frame MP4 animating phase 0→1 with axis limits ±1.
//
// Options:
//   --model=FILE          Model JSON file (required)
//   --nphase=N            Number of frames / phase steps      [300]
//   --phase1=F            Starting phase                      [0.0]
//   --phase2=F            Ending phase                        [1.0]
//   --device=DEV          Gnuplot terminal for live mode      [qt]
//   --x1=F                Left   x-axis limit                 [-1.0]
//   --x2=F                Right  x-axis limit                 [ 1.0]
//   --y1=F                Bottom y-axis limit                 [-1.0]
//   --y2=F                Top    y-axis limit                 [ 1.0]
//   --width=F             Plot width  (pixels if >=64, else inches*100) [8.0]
//   --height=F            Plot height (same logic; default: derived from aspect ratio)
//   --reverse             Dark background colour scheme       [on]
//   --no-reverse          Light background colour scheme
//   --sdOB                Use sdOB colour mapping             [off]
//   --video[=gif|mp4]     Produce a file (default: mp4)
//   --no-video            Live interactive window instead
//   --fps=N               Frame rate for video                [25]
//   --out=FILE            Output filename                     [orbit.mp4 / orbit.gif]
//   --help, -h            Show this help

#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gnuplot-iostream.h>
#include <nlohmann/json.hpp>

#include "../src/lcurve_base/lcurve.h"
#include "../src/new_subs.h"
#include "../src/new_helpers.h"
#include "../src/lroche_base/roche.h"

using namespace std;
namespace C = std::numbers;

template<typename T>
T as(const char* txt, string_view what)
{
    T value{};
    auto [p, ec] = std::from_chars(txt, txt + std::strlen(txt), value);
    if (ec != std::errc{} || p != txt + std::strlen(txt))
        throw std::runtime_error("Cannot parse " + string(what) + " value \"" + txt + '"');
    return value;
}

void plot_visible(Gnuplot& gp,
                  const vector<Lcurve::Point>& object,
                  const Subs::Vec3& earth,
                  const Subs::Vec3& cofm,
                  const Subs::Vec3& xsky,
                  const Subs::Vec3& ysky,
                  double phase)
{
    vector<pair<double,double>> pts;
    pts.reserve(object.size());

    for (auto const& p : object)
        if (Subs::dot(earth, p.dirn) > 0.0 && p.visible(phase))
        {
            Subs::Vec3 r = p.posn - cofm;
            pts.emplace_back(Subs::dot(r, xsky), Subs::dot(r, ysky));
        }

    gp.send1d(pts);
}

struct Cli {
    std::string model;
    int     nphase  = 300;
    double  phase1  = 0.0;
    double  phase2  = 1.0;
    std::string device = "qt";
    double  x1      = -1.0;
    double  x2      =  1.0;
    double  y1      = -1.0;
    double  y2      =  1.0;
    double  width   = 8.0;
    double  height  = -1.0;
    bool    reverse = true;
    bool    sdOB    = false;
    bool    video   = true;
    std::string vformat = "mp4";
    int     fps     = 25;
    std::string out;
    bool    bare    = false;
};

static void print_usage(const char* prog)
{
    cerr <<
        "Usage:\n"
        "  " << prog << " --model=FILE [options...]\n"
        "\n"
        "Only --model is required. Defaults produce a 300-frame MP4 (phase 0→1, ±1 axes).\n"
        "\n"
        "Options:\n"
        "  --model=FILE          Model JSON file (required)\n"
        "  --nphase=N            Number of frames / phase steps      [300]\n"
        "  --phase1=F            Starting phase                      [0.0]\n"
        "  --phase2=F            Ending phase                        [1.0]\n"
        "  --device=DEV          Gnuplot terminal for live mode      [qt]\n"
        "  --x1=F                Left   x-axis limit                 [-1.0]\n"
        "  --x2=F                Right  x-axis limit                 [ 1.0]\n"
        "  --y1=F                Bottom y-axis limit                 [-1.0]\n"
        "  --y2=F                Top    y-axis limit                 [ 1.0]\n"
        "  --width=F             Plot width  (px if >=64, else in*100) [8.0]\n"
        "  --height=F            Plot height (same logic; default: from aspect ratio)\n"
        "  --bare                Hide axes, tics, borders, title    [off]\n"
        "  --no-bare             Show axes and decorations\n"
        "  --reverse             Dark background colour scheme       [on]\n"
        "  --no-reverse          Light background colour scheme\n"
        "  --sdOB                Use sdOB colour mapping             [off]\n"
        "  --video[=gif|mp4]     Produce a file                      [mp4]\n"
        "  --no-video            Live interactive window instead\n"
        "  --fps=N               Frame rate for video                [25]\n"
        "  --out=FILE            Output filename                     [orbit.mp4/gif]\n"
        "  --help, -h            Show this help\n";
}

static std::string shell_quote(const std::string& s)
{
#ifdef _WIN32
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else out.push_back(c);
    }
    out.push_back('"');
    return out;
#else
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') out += "'\"'\"'";
        else out.push_back(c);
    }
    out.push_back('\'');
    return out;
#endif
}

// Extract the value part after '=' in "--key=value", or empty if no '='
static std::string opt_value(const std::string& arg)
{
    auto eq = arg.find('=');
    if (eq == std::string::npos) return {};
    return arg.substr(eq + 1);
}

int main(int argc, char* argv[]) try
{
    using json = nlohmann::json;
    const char* prog = argv[0];

    Cli cli;

    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);

        if (a == "--help" || a == "-h") {
            print_usage(prog);
            return EXIT_SUCCESS;
        }
        else if (a.rfind("--model=", 0) == 0)   { cli.model   = opt_value(a); }
        else if (a.rfind("--nphase=", 0) == 0)   { cli.nphase  = as<int>(opt_value(a).c_str(), "nphase"); }
        else if (a.rfind("--phase1=", 0) == 0)   { cli.phase1  = as<double>(opt_value(a).c_str(), "phase1"); }
        else if (a.rfind("--phase2=", 0) == 0)   { cli.phase2  = as<double>(opt_value(a).c_str(), "phase2"); }
        else if (a.rfind("--device=", 0) == 0)   { cli.device  = opt_value(a); }
        else if (a.rfind("--x1=", 0) == 0)       { cli.x1      = as<double>(opt_value(a).c_str(), "x1"); }
        else if (a.rfind("--x2=", 0) == 0)       { cli.x2      = as<double>(opt_value(a).c_str(), "x2"); }
        else if (a.rfind("--y1=", 0) == 0)       { cli.y1      = as<double>(opt_value(a).c_str(), "y1"); }
        else if (a.rfind("--y2=", 0) == 0)       { cli.y2      = as<double>(opt_value(a).c_str(), "y2"); }
        else if (a.rfind("--width=", 0) == 0)    { cli.width   = as<double>(opt_value(a).c_str(), "width"); }
        else if (a.rfind("--height=", 0) == 0)   { cli.height  = as<double>(opt_value(a).c_str(), "height"); }
        else if (a == "--bare")                   { cli.bare = true; }
        else if (a == "--no-bare")                { cli.bare = false; }
        else if (a == "--reverse")                { cli.reverse = true; }
        else if (a == "--no-reverse")             { cli.reverse = false; }
        else if (a == "--sdOB")                   { cli.sdOB    = true; }
        else if (a == "--video")                  { cli.video   = true; cli.vformat = "mp4"; }
        else if (a.rfind("--video=", 0) == 0) {
            cli.video = true;
            std::string v = opt_value(a);
            if (v == "gif" || v == "GIF")      cli.vformat = "gif";
            else if (v == "mp4" || v == "MP4") cli.vformat = "mp4";
            else { cerr << "Unknown --video format '" << v << "' (use gif or mp4)\n"; return EXIT_FAILURE; }
        }
        else if (a == "--no-video")               { cli.video = false; }
        else if (a.rfind("--fps=", 0) == 0)      { cli.fps = std::clamp(as<int>(opt_value(a).c_str(), "fps"), 1, 240); }
        else if (a.rfind("--out=", 0) == 0)       { cli.out = opt_value(a); }
        else {
            cerr << "Unknown option: " << a << "\n";
            print_usage(prog);
            return EXIT_FAILURE;
        }
    }

    if (cli.model.empty()) {
        cerr << "Error: --model=FILE is required.\n\n";
        print_usage(prog);
        return EXIT_FAILURE;
    }

    if (cli.out.empty())
        cli.out = (cli.vformat == "gif") ? "orbit.gif" : "orbit.mp4";

    auto width_to_pixels = [](double w) -> int {
        if (w < 64.0) return static_cast<int>(std::round(w * 100.0));
        return static_cast<int>(std::round(w));
    };

    const int Wpx = width_to_pixels(cli.width) & ~1;
    const int Hpx = ((cli.height > 0.0)
                        ? width_to_pixels(cli.height)
                        : static_cast<int>(std::round(Wpx * (cli.y2 - cli.y1) / (cli.x2 - cli.x1))))
                     & ~1;

    string config_file = cli.model;
    auto model_config = Helpers::load_model_and_config_from_json(config_file);
    Lcurve::Model model = model_config.first;
    json config = model_config.second;

    double r1{}, r2{};
    model.get_r1r2(r1, r2);

    const double rl1 = Roche::xl11(model.q, model.spin1);
    const double rl2 = 1.0 - Roche::xl12(model.q, model.spin2);

    if (r1 <= 0.0) r1 = 0.999999999999 * rl1;
    if (r2 <= 0.0) r2 = 0.999999999999 * rl2;

    vector<Lcurve::Point> star1, star2, disc, outer_edge, inner_edge, stream;

    Lcurve::set_star_grid(model, Roche::PRIMARY,   true, star1);
    Lcurve::set_star_grid(model, Roche::SECONDARY, true, star2);

    double rdisc1 = 0.0, rdisc2 = 0.0;
    if (model.add_disc)
    {
        rdisc1 = (model.rdisc1 > 0.0) ? model.rdisc1 : r1;
        rdisc2 = (model.rdisc2 > 0.0) ? model.rdisc2 : model.radius_spot;

        Lcurve::set_disc_grid(model,      disc);
        Lcurve::set_disc_edge(model,true, outer_edge);
        Lcurve::set_disc_edge(model,false,inner_edge);

        if (model.opaque)
        {
            for (auto* obj : {&star1, &star2})
                for (auto& pt : *obj)
                    for (auto& e : Roche::disc_eclipse(model.iangle, rdisc1, rdisc2,
                                                       model.beta_disc, model.height_disc, pt.posn))
                        pt.eclipse.push_back(e);
        }
    }

    if (model.add_spot)
    {
        Subs::Vec3 dir(1,0,0), posn, v;
        Roche::strinit(model.q, posn, v);

        const double rl1_ = Roche::xl1(model.q);

        auto push_stream_point = [&](Subs::Vec3 const& P)
        {
            Lcurve::Point::etype eclipses;
            for (auto& e : Roche::disc_eclipse(model.iangle, rdisc1, rdisc2,
                                               model.beta_disc, model.height_disc, P))
                eclipses.push_back(e);
            stream.emplace_back(P, dir, 0.0, 1.0, eclipses);
        };

        push_stream_point(posn);

        constexpr int NSTREAM = 200;
        for (int i=0;i<NSTREAM;++i)
        {
            double radius = rl1_ + (model.radius_spot - rl1_) * (i+1) / NSTREAM;
            Roche::stradv(model.q, posn, v, radius, 1e-10, 1e-3);
            push_stream_point(posn);
        }
    }

    Gnuplot gp("gnuplot 2>/dev/null");

    const int delay_cs = std::max(1, 100 / std::max(1, cli.fps));

    if (cli.video) {
        if (cli.vformat == "gif") {
            gp << "set term gif animate optimize size " << Wpx << "," << Hpx
               << " delay " << delay_cs << "\n";
            gp << "set output " << std::quoted(cli.out) << "\n";
        } else {
            gp << "set term pngcairo size " << Wpx << "," << Hpx << "\n";
        }
    } else {
        double live_w = cli.width;
        double live_h = (cli.height > 0.0) ? cli.height
                         : cli.width * (cli.y2 - cli.y1) / (cli.x2 - cli.x1);
        gp << "set term " << cli.device << " size " << live_w << "," << live_h << "\n";
    }

    gp << "unset key\n";
    double aspect = (cli.y2 - cli.y1) / (cli.x2 - cli.x1);
    gp << "set size ratio " << aspect << "\n";
    gp << "set xrange [" << cli.x1 << ":" << cli.x2 << "]\n";
    gp << "set yrange [" << cli.y1 << ":" << cli.y2 << "]\n";

    if (cli.bare) {
        gp << "unset border\n";
        gp << "unset tics\n";
        gp << "unset xlabel\n";
        gp << "unset ylabel\n";
        gp << "unset title\n";
        gp << "unset zeroaxis\n";
        gp << "unset grid\n";
        gp << "set lmargin 0\n";
        gp << "set rmargin 0\n";
        gp << "set tmargin 0\n";
        gp << "set bmargin 0\n";
    } else {
        gp << "set zeroaxis\n";
        gp << "set tmargin at screen 0.95\n";
        gp << "set bmargin at screen 0.05\n";
    }
    
    auto colour = [&](int which)
    {
        if (!cli.reverse)
            switch(which){ case 0:return "black"; case 2:return cli.sdOB?"blue":"red";
                            case 4:return cli.sdOB?"red":"blue"; default:return "grey"; }
        else
            switch(which){ case 0:return "white"; case 1:return "black";
                            case 2:return cli.sdOB?"#000080":"#660000";
                            case 3:return "#005000";
                            case 4:return cli.sdOB?"#660000":"#000080";
                            default:return "grey"; }
    };

    auto add_spec = [&](const std::string& clr, bool &first){
        if (first) {
            gp << "plot ";
            first = false;
        } else {
            gp << ", ";
        }
        gp << "'-' w p pt 7 lc rgb '" << clr << "' notitle";
    };

    const Subs::Vec3 cofm(model.q/(1.+model.q), 0., 0.);

    std::filesystem::path tmpdir;
    if (cli.video && cli.vformat == "mp4") {
        tmpdir = std::filesystem::temp_directory_path() /
                 std::format("visualise_frames_{:x}", std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::filesystem::create_directories(tmpdir);
    }

    for (int ip = 0; ip < cli.nphase; ++ip)
    {
        const double phase =
            (cli.nphase == 1) ? cli.phase1
                              : (cli.phase1 + (cli.phase2 - cli.phase1) * ip / double(cli.nphase - 1));

        const double cosp = std::cos(C::pi * 2 * phase);
        const double sinp = std::sin(C::pi * 2 * phase);

        Subs::Vec3 earth = Roche::set_earth(model.iangle, phase);
        Subs::Vec3 xsky(sinp,  cosp, 0.);
        Subs::Vec3 ysky = Subs::cross(earth, xsky);

        if (cli.video && cli.vformat == "mp4") {
            auto framefile = tmpdir / std::format("frame_{:05}.png", ip);
            gp << "set output " << std::quoted(framefile.string()) << "\n";
        }

        if (!cli.video)
            gp << "clear\n";

        if (!cli.bare)
            gp << "set title " << std::quoted(std::format("phase={:.4f}", phase)) << '\n';

        // Depth-sort stars: farther one drawn first, closer one drawn last
        bool star1_in_front = Subs::dot(Subs::Vec3(0,0,0), earth)
                            > Subs::dot(Subs::Vec3(1,0,0), earth);

        auto& back_star  = star1_in_front ? star2 : star1;
        auto& front_star = star1_in_front ? star1 : star2;
        const char* back_clr  = star1_in_front ? colour(2) : colour(4);
        const char* front_clr = star1_in_front ? colour(4) : colour(2);

        bool first = true;
        add_spec(back_clr, first);
        if (model.add_disc) {
            add_spec(colour(3), first);
            add_spec(colour(1), first);
            add_spec(colour(1), first);
        }
        if (model.add_spot) {
            add_spec(colour(2), first);
        }
        add_spec(front_clr, first);
        gp << std::endl;

        plot_visible(gp, back_star, earth, cofm, xsky, ysky, phase);
        if (model.add_disc) {
            plot_visible(gp, disc,       earth, cofm, xsky, ysky, phase);
            plot_visible(gp, outer_edge, earth, cofm, xsky, ysky, phase);
            plot_visible(gp, inner_edge, earth, cofm, xsky, ysky, phase);
        }
        if (model.add_spot) {
            plot_visible(gp, stream, earth, cofm, xsky, ysky, phase);
        }
        plot_visible(gp, front_star, earth, cofm, xsky, ysky, phase);

        gp.flush();

        if (cli.video && cli.vformat == "mp4") {
            gp << "unset output\n";
            gp.flush();

            int done = ip + 1;
            int total = cli.nphase;
            int barw = 40;
            int filled = (done * barw) / total;
            cerr << "\r[";
            for (int b = 0; b < barw; ++b)
                cerr << (b < filled ? '#' : '.');
            cerr << "] " << done << "/" << total
                    << " (" << (done * 100 / total) << "%)";
            if (done == total) cerr << '\n';
            cerr << std::flush;
        }
        if (!cli.video) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    if (cli.video) {
        if (cli.vformat == "gif") {
            gp << "unset output\n";
            gp.flush();
            cerr << "Wrote " << cli.out << '\n';
        } else {
            gp << "unset output\n";
            gp.flush();

            std::string pattern = (tmpdir / "frame_%05d.png").string();
            std::string cmd =
                "ffmpeg -y -hide_banner -v error -xerror "
                "-framerate " + std::to_string(cli.fps) +
                " -f image2 -start_number 0 -i " + shell_quote(pattern) +
                " -c:v libx264 -pix_fmt yuv420p -movflags +faststart " + shell_quote(cli.out);

            int rc = std::system(cmd.c_str());
            if (rc != 0) {
                cerr << "ffmpeg failed (" << rc << "). Command:\n" << cmd << "\n";
                cerr << "Keeping frames in: " << tmpdir << "\n";
            }
            else {
                cerr << "Wrote " << cli.out << '\n';
            }
            try {
                for (auto& e : std::filesystem::directory_iterator(tmpdir))
                    std::filesystem::remove(e.path());
                std::filesystem::remove(tmpdir);
            } catch (...) {}
        }
    }

    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    cerr << "visualize: " << e.what() << '\n';
    return EXIT_FAILURE;
}