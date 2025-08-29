// visualize.cpp
// Build an interactive plot or a video (GIF/MP4) of the system using gnuplot.
// Usage (single phase):
//   ./visualize [--video[=gif|mp4]] [--fps=N] [--out=FILE]
//       model nphase(=1) phase device x1 x2 y1 y2 [width] [reverse] [sdOB]
// Usage (phase range):
//   ./visualize [--video[=gif|mp4]] [--fps=N] [--out=FILE]
//       model nphase(>1) phase1 phase2 device x1 x2 y1 y2 [width] [reverse] [sdOB]
//
// Notes:
// - --video produces a file instead of a live window. Default format is GIF.
// - --fps sets frame rate for GIF/MP4. Default 25.
// - --out sets output filename; defaults: orbit.gif / orbit.mp4.
// - Width semantics:
//     * Live mode: gnuplot 'size' uses your terminal's units (often pixels for qt/x11).
//     * Video mode: 'width' is treated as pixels if >=64; if smaller (e.g. default 8.0),
//       it is interpreted as "inches*100" so 8.0 -> 800 pixels.

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

//------------------------------------------------------------------------------
// helper: convert argv string to any arithmetic type
//------------------------------------------------------------------------------
template<typename T>
T as(const char* txt, string_view what)
{
    T value{};
    auto [p, ec] = std::from_chars(txt, txt + std::strlen(txt), value);
    if (ec != std::errc{} || p != txt + std::strlen(txt))
        throw std::runtime_error("Cannot parse " + string(what) + " value \"" + txt + '"');
    return value;
}

//------------------------------------------------------------------------------
// take the object grids (star, disc, stream) and plot only points that are
// visible at the requested orbital phase
//------------------------------------------------------------------------------
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
    bool video = false;            // produce a file instead of live plot
    std::string vformat = "gif";   // "gif" or "mp4"
    int fps = 25;                  // frames per second
    std::string out;               // output filename; default by format
};

static void print_usage(const char* prog)
{
    cerr <<
        "Usage (single phase):\n"
        "  " << prog << " [--video[=gif|mp4]] [--fps=N] [--out=FILE]\n"
        "      model nphase(=1) phase device x1 x2 y1 y2 [width] [reverse] [sdOB]\n"
        "Usage (phase range):\n"
        "  " << prog << " [--video[=gif|mp4]] [--fps=N] [--out=FILE]\n"
        "      model nphase(>1) phase1 phase2 device x1 x2 y1 y2 [width] [reverse] [sdOB]\n";
}

// Shell-safe quoting for system() command
static std::string shell_quote(const std::string& s)
{
#ifdef _WIN32
    // Simple CMD quoting: wrap in double quotes and escape internal quotes
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
    // POSIX: single-quote, escape single quotes by closing/opening with '"'"'
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

//------------------------------------------------------------------------------
// main
//------------------------------------------------------------------------------
int main(int argc, char* argv[]) try
{
    using json = nlohmann::json;
    const char* prog = argv[0];

    // Parse options first, collect positional args in 'pos'
    Cli cli;
    std::vector<std::string> pos;
    pos.reserve(argc);
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a.rfind("--", 0) == 0) {
            if (a == "--video") {
                cli.video = true;
            } else if (a.rfind("--video=", 0) == 0) {
                cli.video = true;
                std::string v = a.substr(8);
                if (v == "gif" || v == "GIF") cli.vformat = "gif";
                else if (v == "mp4" || v == "MP4") cli.vformat = "mp4";
                else {
                    cerr << "Unknown --video format '" << v << "' (use gif or mp4)\n";
                    return EXIT_FAILURE;
                }
            } else if (a.rfind("--fps=", 0) == 0) {
                int f = std::max(1, std::min(240, as<int>(a.c_str() + 6, "fps")));
                cli.fps = f;
            } else if (a.rfind("--out=", 0) == 0) {
                cli.out = a.substr(6);
            } else if (a == "--help" || a == "-h") {
                print_usage(prog);
                return EXIT_SUCCESS;
            } else {
                // Unrecognized option -> treat as positional to avoid breaking old scripts
                pos.push_back(a);
            }
        } else {
            pos.push_back(std::move(a));
        }
    }

    if (pos.size() < 8) { // minimal: model nphase phase device x1 x2 y1 y2
        print_usage(prog);
        return EXIT_FAILURE;
    }

    //--- positional arguments --------------------------------------------------
    size_t p = 0;
    string  smodel   = pos[p++];
    int     nphase   = as<int>(pos[p++].c_str(), "nphase");
    double  phase1   = as<double>(pos[p++].c_str(), "phase/phase1");
    double  phase2   = (nphase == 1) ? phase1 : (p < pos.size() ? as<double>(pos[p++].c_str(), "phase2")
                                                                : throw std::runtime_error("Missing phase2 for nphase>1"));
    string  device   = pos[p++];
    double  x1       = as<double>(pos[p++].c_str(), "x1");
    double  x2       = as<double>(pos[p++].c_str(), "x2");
    double  y1       = as<double>(pos[p++].c_str(), "y1");
    double  y2       = as<double>(pos[p++].c_str(), "y2");
    double  width    = (p < pos.size()) ? as<double>(pos[p++].c_str(), "width") : 8.0;
    bool    reverse  = (p < pos.size()) ? bool(as<int>(pos[p++].c_str(), "reverse")) : true;
    bool    sdOB     = (p < pos.size()) ? bool(as<int>(pos[p++].c_str(), "sdOB"))    : false;

    // Defaults for output name
    if (cli.out.empty()) {
        cli.out = (cli.vformat == "gif") ? "orbit.gif" : "orbit.mp4";
    }

    string config_file = smodel;
    auto model_config = Helpers::load_model_and_config_from_json(config_file);
    Lcurve::Model model = model_config.first;
    json config = model_config.second;

    double r1{}, r2{};
    model.get_r1r2(r1, r2);

    const double rl1 = Roche::xl11(model.q, model.spin1);
    const double rl2 = 1.0 - Roche::xl12(model.q, model.spin2);

    if (r1 <= 0.0) r1 = 0.999999999999 * rl1;
    if (r2 <= 0.0) r2 = 0.999999999999 * rl2;

    //---------------------------------------------------------------------------
    // Produce point grids ------------------------------------------------------
    //---------------------------------------------------------------------------
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

        push_stream_point(posn);                            // anchor point

        constexpr int NSTREAM = 200;
        for (int i=0;i<NSTREAM;++i)
        {
            double radius = rl1_ + (model.radius_spot - rl1_) * (i+1) / NSTREAM;
            Roche::stradv(model.q, posn, v, radius, 1e-10, 1e-3);
            push_stream_point(posn);
        }
    }

    //---------------------------------------------------------------------------
    // Gnuplot setup ------------------------------------------------------------
    //---------------------------------------------------------------------------
    Gnuplot gp;

    auto width_to_pixels = [&](double w)->int {
        // If the given width is small (e.g. default 8.0), treat as ~ inches*100 => 800 px
        if (w < 64.0) return static_cast<int>(std::round(w * 100.0));
        return static_cast<int>(std::round(w));
    };

    const int Wpx = width_to_pixels(width);
    const int Hpx = static_cast<int>(std::round(Wpx * (y2 - y1) / (x2 - x1)));
    const int delay_cs = std::max(1, 100 / std::max(1, cli.fps));  // GIF delay in 1/100 s

    if (cli.video) {
        if (cli.vformat == "gif") {
            gp << "set term gif animate optimize size " << Wpx << "," << Hpx
               << " delay " << delay_cs << "\n";
            gp << "set output " << std::quoted(cli.out) << "\n";
        } else { // mp4 via png frames + ffmpeg
            gp << "set term pngcairo size " << Wpx << "," << Hpx << "\n";
        }
    } else {
        gp << "set term " << device << " size " << width
           << "," << width * (y2 - y1) / (x2 - x1) << "\n";
    }

    gp << "unset key\n";
    gp << "set size square\n";
    gp << "set xrange [" << x1 << ":" << x2 << "]\n";
    gp << "set yrange [" << y1 << ":" << y2 << "]\n";
    gp << "set zeroaxis\n";
    gp << "set tmargin at screen 0.95\n";
    gp << "set bmargin at screen 0.05\n";

    // colour palette (resembles the old pgplot selection)
    auto colour = [&](int which)
    {
        if (!reverse)
            switch(which){ case 0:return "black"; case 2:return sdOB?"blue":"red";
                            case 4:return sdOB?"red":"blue"; default:return "grey"; }
        else               // reversed
            switch(which){ case 0:return "white"; case 1:return "black";
                            case 2:return sdOB?"#000080":"#660000";
                            case 3:return "#005000";          // dark green
                            case 4:return sdOB?"#660000":"#000080";
                            default:return "grey"; }
    };

    // Helper: add only the plot spec (no newline) for a dataset using inline data '-'
    auto add_spec = [&](const std::string& clr, bool &first){
        if (first) {
            gp << "plot ";
            first = false;
        } else {
            gp << ", ";
        }
        gp << "'-' w p pt 7 lc rgb '" << clr << "' notitle";
    };

    //---------------------------------------------------------------------------
    // orbital loop -------------------------------------------------------------
    //---------------------------------------------------------------------------
    const Subs::Vec3 cofm(model.q/(1.+model.q), 0., 0.);

    // For mp4 we will output PNG frames then run ffmpeg
    std::filesystem::path tmpdir;
    if (cli.video && cli.vformat == "mp4") {
        tmpdir = std::filesystem::temp_directory_path() /
                 std::format("visualise_frames_{:x}", std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::filesystem::create_directories(tmpdir);
    }

    for (int ip = 0; ip < nphase; ++ip)
    {
        const double phase =
            (nphase == 1) ? phase1
                          : (phase1 + (phase2 - phase1) * ip / double(nphase - 1));

        // Basis vectors on the sky
        const double cosp = std::cos(C::pi * 2 * phase);
        const double sinp = std::sin(C::pi * 2 * phase);

        Subs::Vec3 earth = Roche::set_earth(model.iangle, phase);   // points towards observer
        Subs::Vec3 xsky(sinp,  cosp, 0.);
        Subs::Vec3 ysky = Subs::cross(earth, xsky);

        gp << "clear\n";     // <-- keep it if you want to reset the state

        if (cli.video && cli.vformat == "mp4") {
            auto framefile = tmpdir / std::format("frame_{:05}.png", ip);
            gp << "set output " << std::quoted(framefile.string()) << "\n";
        }
        gp << "set title " << std::quoted(std::format("phase={:.4f}", phase)) << '\n';

        // 1) Write the whole plot command (all '-' specs), but DO NOT send data yet
        bool first = true;
        add_spec(colour(4), first);                 // star1
        add_spec(colour(2), first);                 // star2
        if (model.add_disc) {
            add_spec(colour(3), first);             // disc
            add_spec(colour(1), first);             // outer_edge
            add_spec(colour(1), first);             // inner_edge
        }
        if (model.add_spot) {
            add_spec(colour(2), first);             // stream
        }
        gp << std::endl; // end plot command (gnuplot now expects data blocks)

        // 2) Now send the data blocks in the exact same order as the '-' specs
        plot_visible(gp, star1, earth, cofm, xsky, ysky, phase);
        plot_visible(gp, star2, earth, cofm, xsky, ysky, phase);
        if (model.add_disc) {
            plot_visible(gp, disc,       earth, cofm, xsky, ysky, phase);
            plot_visible(gp, outer_edge, earth, cofm, xsky, ysky, phase);
            plot_visible(gp, inner_edge, earth, cofm, xsky, ysky, phase);
        }
        if (model.add_spot) {
            plot_visible(gp, stream, earth, cofm, xsky, ysky, phase);
        }

        gp.flush();  // finish this frame

        if (cli.video && cli.vformat == "mp4") {
            gp << "unset output\n";
            gp.flush();
        }
        if (!cli.video) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));   // crude animation live
        }
    }

    // Finalize outputs
    if (cli.video) {
        if (cli.vformat == "gif") {
            gp << "unset output\n";
            gp.flush();
            cerr << "Wrote " << cli.out << '\n';
        } else {
            // MP4 via ffmpeg
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
            // Cleanup temporary frames
            try {
                for (auto& e : std::filesystem::directory_iterator(tmpdir)){
                        std::filesystem::remove(e.path());
                        std::filesystem::remove(tmpdir);
                    }
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