// report_writer.h
// TeX/PDF fit summary report generator for LCURVE
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include "physical_prior.h"

namespace ReportWriter {

// ─────────────────────────────────────────────────────────────────────
struct FitReportData {
    std::string config_path;

    bool        converged = false;
    std::string stop_reason;
    int         iterations = 0, func_evals = 0;
    double      chisq_lc = 0, sum_sq = 0;
    int         ndata = 0, npar = 0, nprior = 0;

    std::vector<std::string>              par_names;
    std::vector<double>                   par_values;
    std::vector<double>                   par_sigmas;
    std::vector<std::pair<double,double>> par_limits;

    DerivedQuantities    derived;
    ObservedConstraints  obs;
    bool has_priors     = false;
    bool has_covariance = false;

    std::vector<std::vector<double>> correlations;

    std::vector<std::string> prior_names;
    std::vector<double>      prior_residuals;
};

// ─────────────────────────────────────────────────────────────────────
inline std::string tex_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '_': o += "\\_"; break;
            case '%': o += "\\%"; break;
            case '&': o += "\\&"; break;
            case '#': o += "\\#"; break;
            case '$': o += "\\$"; break;
            case '{': o += "\\{"; break;
            case '}': o += "\\}"; break;
            case '~': o += "\\textasciitilde{}"; break;
            case '^': o += "\\textasciicircum{}"; break;
            default:  o += c;
        }
    }
    return o;
}

inline std::string param_to_latex(const std::string& n) {
    if (n == "q")              return "$q$";
    if (n == "iangle")         return "$i$~(deg)";
    if (n == "r1")             return "$r_1$";
    if (n == "r2")             return "$r_2$";
    if (n == "cphi3")          return "$\\cos\\phi_3$";
    if (n == "cphi4")          return "$\\cos\\phi_4$";
    if (n == "spin1")          return "$F_1$";
    if (n == "spin2")          return "$F_2$";
    if (n == "t1")             return "$T_1$~(K)";
    if (n == "t2")             return "$T_2$~(K)";
    if (n == "velocity_scale") return "$v_{\\rm scale}$~(km/s)";
    if (n == "beam_factor1")   return "$B_1$";
    if (n == "beam_factor2")   return "$B_2$";
    if (n == "tperiod")        return "$P$~(d)";
    if (n == "period")         return "$P$~(d)";
    if (n == "gdark_bolom1")   return "$\\beta_1$";
    if (n == "gdark_bolom2")   return "$\\beta_2$";
    if (n == "absorb")         return "$A_{\\rm abs}$";
    if (n == "slope")          return "slope";
    if (n == "quad")           return "quad";
    if (n == "cube")           return "cube";
    if (n == "third")          return "$\\ell_3$";
    if (n == "ldc1_1")         return "$u_{1,1}$";
    if (n == "ldc1_2")         return "$u_{1,2}$";
    if (n == "ldc1_3")         return "$u_{1,3}$";
    if (n == "ldc1_4")         return "$u_{1,4}$";
    if (n == "ldc2_1")         return "$u_{2,1}$";
    if (n == "ldc2_2")         return "$u_{2,2}$";
    if (n == "ldc2_3")         return "$u_{2,3}$";
    if (n == "ldc2_4")         return "$u_{2,4}$";
    if (n == "phase_offset")   return "$\\Delta\\phi$";
    if (n == "delta_phase")    return "$\\delta\\phi$";
    if (n == "gravity_dark1")  return "$g_1$";
    if (n == "gravity_dark2")  return "$g_2$";
    if (n == "t0")             return "$T_0$";
    return tex_escape(n);
}

inline std::string param_to_latex_short(const std::string& n) {
    if (n == "q")              return "$q$";
    if (n == "iangle")         return "$i$";
    if (n == "r1")             return "$r_1$";
    if (n == "r2")             return "$r_2$";
    if (n == "cphi3")          return "$c_3$";
    if (n == "cphi4")          return "$c_4$";
    if (n == "spin1")          return "$F_1$";
    if (n == "spin2")          return "$F_2$";
    if (n == "t1")             return "$T_1$";
    if (n == "t2")             return "$T_2$";
    if (n == "velocity_scale") return "$v_s$";
    if (n == "beam_factor1")   return "$B_1$";
    if (n == "beam_factor2")   return "$B_2$";
    if (n == "tperiod")        return "$P$";
    if (n == "period")         return "$P$";
    if (n == "gdark_bolom1")   return "$\\beta_1$";
    if (n == "gdark_bolom2")   return "$\\beta_2$";
    if (n == "third")          return "$\\ell_3$";
    if (n == "ldc1_1")         return "$u_{11}$";
    if (n == "ldc1_2")         return "$u_{12}$";
    if (n == "ldc2_1")         return "$u_{21}$";
    if (n == "ldc2_2")         return "$u_{22}$";
    return param_to_latex(n);
}

inline std::string tex_val_pm(double val, double err) {
    if (err <= 0.0) {
        std::ostringstream o;
        o << std::fixed << std::setprecision(4) << val;
        return "$" + o.str() + "$";
    }
    int eexp = (std::abs(err) > 0)
             ? (int)std::floor(std::log10(std::abs(err))) : 0;
    int prec = std::clamp(1 - eexp, 0, 8);
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec)
      << "$" << val << " \\pm " << err << "$";
    return o.str();
}

inline std::string tex_obs(double val, double lo, double hi) {
    int eref = std::max(lo, hi) > 0
             ? (int)std::floor(std::log10(std::max(lo, hi))) : 0;
    int prec = std::clamp(1 - eref, 0, 6);
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec)
      << "$" << val << "^{+" << hi << "}_{-" << lo << "}$";
    return o.str();
}

inline std::string derive_report_path(const std::string& config_path) {
    std::string base = config_path;
    auto pos = base.rfind(".json");
    if (pos != std::string::npos && pos == base.size() - 5)
        base = base.substr(0, pos);
    return base + "_report.tex";
}

// ─────────────────────────────────────────────────────────────────────
inline double compute_pull(double implied, double obs_val,
                           double err_lo, double err_hi) {
    double diff  = implied - obs_val;
    double sigma = (diff >= 0) ? err_hi : err_lo;
    if (sigma <= 0) sigma = std::max(err_lo, err_hi);
    if (sigma <= 0) return 0.0;
    return diff / sigma;
}

// ─────────────────────────────────────────────────────────────────────
inline void write_tex_report(const FitReportData& rd,
                             const std::string& tex_path_in = "")
{
    std::string tex_path = tex_path_in.empty()
                         ? derive_report_path(rd.config_path)
                         : tex_path_in;

    std::ofstream out(tex_path);
    if (!out.is_open()) {
        std::cerr << "Could not open " << tex_path << " for writing.\n";
        return;
    }

    double red_chi2 = rd.chisq_lc / std::max(1, rd.ndata - rd.npar);

    // ── preamble ──
    out << "\\documentclass[11pt,a4paper]{article}\n"
        << "\\usepackage[margin=2.2cm]{geometry}\n"
        << "\\usepackage{booktabs}\n"
        << "\\usepackage{amsmath}\n"
        << "\\usepackage[table]{xcolor}\n"
        << "\\usepackage{hyperref}\n"
        << "\\usepackage{colortbl}\n"
        << "\\usepackage{array}\n"
        << "\\usepackage{tabularx}\n"
        << "\\usepackage{tcolorbox}\n"
        << "\\tcbuselibrary{skins}\n\n"

        << "\\hypersetup{colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!60!black}\n\n"

        // Tighter row spacing control
        << "\\renewcommand{\\arraystretch}{1.25}\n"
        // Alternating row color
        << "\\definecolor{rowA}{gray}{0.95}\n"
        << "\\definecolor{rowB}{gray}{1.00}\n"
        << "\\definecolor{headerblue}{HTML}{2C3E6B}\n"
        << "\\definecolor{headerfg}{HTML}{FFFFFF}\n"
        << "\\definecolor{accentblue}{HTML}{3B7DD8}\n"
        << "\\definecolor{framegray}{HTML}{CCCCCC}\n"
        << "\\definecolor{lightblue}{HTML}{EBF2FC}\n\n"

        // Section styling
        << "\\usepackage{titlesec}\n"
        << "\\titleformat{\\section}{\\Large\\bfseries\\color{headerblue}}{\\thesection.}{0.6em}{}\n"
        << "\\titlespacing*{\\section}{0pt}{1.8ex plus .6ex minus .3ex}{1.0ex plus .3ex}\n\n"

        // Header rule helper
        << "\\newcommand{\\thead}[1]{\\textbf{\\color{headerfg}#1}}\n\n"

        << "\\title{\\color{headerblue}\\LARGE Light-Curve Fit Report}\n"
        << "\\author{\\textsc{LCURVE} --- Levenberg--Marquardt Solver\\\\[2pt]\n"
        << "{\\small\\texttt{" << tex_escape(rd.config_path) << "}}}\n"
        << "\\date{\\today}\n\n"
        << "\\begin{document}\n"
        << "\\maketitle\n"
        << "\\thispagestyle{empty}\n\n";

    // ═══════════════ Section 1: Fit Summary ═══════════════════════════
    out << "\\section{Fit Summary}\n\n";

    // Convergence status badge
    if (rd.converged) {
        out << "\\begin{tcolorbox}[colback=green!8,colframe=green!50!black,"
            << "boxrule=0.5pt,arc=3pt,left=6pt,right=6pt,top=4pt,bottom=4pt]\n"
            << "\\textbf{\\color{green!50!black}\\large $\\checkmark$ Converged}"
            << " --- " << tex_escape(rd.stop_reason) << "\n"
            << "\\end{tcolorbox}\n\n";
    } else {
        out << "\\begin{tcolorbox}[colback=red!8,colframe=red!60!black,"
            << "boxrule=0.5pt,arc=3pt,left=6pt,right=6pt,top=4pt,bottom=4pt]\n"
            << "\\textbf{\\color{red!60!black}\\large $\\boldsymbol{\\times}$ Did Not Converge}"
            << " --- " << tex_escape(rd.stop_reason) << "\n"
            << "\\end{tcolorbox}\n\n";
    }

    out << "\\begin{tcolorbox}[colback=lightblue,colframe=framegray,"
        << "boxrule=0.4pt,arc=2pt,left=8pt,right=8pt,top=6pt,bottom=6pt]\n"
        << "\\begin{tabular}{l@{\\hskip 1.5em}r@{\\hskip 3em}l@{\\hskip 1.5em}r}\n";
    {
        std::ostringstream o; o << std::fixed << std::setprecision(4);
        o << rd.chisq_lc;
        std::string chi2_str = o.str();
        o.str(""); o << red_chi2;
        std::string red_chi2_str = o.str();
        o.str(""); o << rd.sum_sq;
        std::string sum_sq_str = o.str();

        out << "Iterations & " << rd.iterations
            << " & $\\chi^2_{\\rm LC}$ & " << chi2_str << " \\\\\n"
            << "Function evaluations & " << rd.func_evals
            << " & $\\chi^2_{\\rm red}$ & " << red_chi2_str << " \\\\\n"
            << "$N_{\\rm data}$ & " << rd.ndata
            << " & $\\|r\\|^2$ & " << sum_sq_str << " \\\\\n"
            << "$N_{\\rm par}$ & " << rd.npar
            << " & $N_{\\rm prior}$ & " << rd.nprior << " \\\\\n";
    }
    if (rd.has_priors) {
        std::ostringstream o;
        o << std::fixed << std::setprecision(8) << rd.obs.P_days;
        out << "$P_{\\rm orb}$ (d) & \\multicolumn{3}{l}{" << o.str() << "} \\\\\n";
    }
    out << "\\end{tabular}\n"
        << "\\end{tcolorbox}\n\n";

    // ═══════════════ Section 2: Best-Fit Parameters ══════════════════
    out << "\\section{Best-Fit Parameters}\n\n"
        << "\\rowcolors{2}{rowA}{rowB}\n"
        << "\\begin{tabular}{l r r}\n"
        << "\\rowcolor{headerblue}\n"
        << "\\thead{Parameter} & \\thead{"
        << (rd.has_covariance ? "Value $\\pm\\sigma$" : "Value")
        << "} & \\thead{Limits} \\\\\n"
        << "\\toprule\n";

    for (int i = 0; i < rd.npar; ++i) {
        out << param_to_latex(rd.par_names[i]) << " & ";
        if (rd.has_covariance && rd.par_sigmas[i] > 0)
            out << tex_val_pm(rd.par_values[i], rd.par_sigmas[i]);
        else {
            std::ostringstream o;
            o << std::fixed << std::setprecision(6) << rd.par_values[i];
            out << "$" << o.str() << "$";
        }
        {
            std::ostringstream o;
            o << std::fixed << std::setprecision(4)
              << " & $[" << rd.par_limits[i].first
              << ",\\;" << rd.par_limits[i].second << "]$";
            out << o.str();
        }
        out << " \\\\\n";
    }
    out << "\\bottomrule\n"
        << "\\end{tabular}\n"
        << "\\rowcolors{0}{}{}\n\n";

    // ═══════════════ Section 3: Derived Quantities ═══════════════════
    if (rd.has_priors) {
        out << "\\section{Derived Physical Quantities}\n\n"
            << "\\rowcolors{2}{rowA}{rowB}\n"
            << "\\begin{tabular}{l r r r}\n"
            << "\\rowcolor{headerblue}\n"
            << "\\thead{Quantity} & \\thead{Implied"
            << (rd.derived.has_errors ? " $\\pm\\sigma$" : "")
            << "} & \\thead{Observed} & \\thead{Pull} \\\\\n"
            << "\\toprule\n";

        const auto& dq = rd.derived;
        const auto& ob = rd.obs;

        struct Row {
            std::string label;
            double impl, impl_err;
            bool   has_obs;
            double obs_v, obs_lo, obs_hi;
        };

        std::vector<Row> rows = {
            {"$K_1$ (km/s)",     dq.K1,      dq.K1_err,
             ob.has_K,  ob.K_obs,  ob.K_err_lo,  ob.K_err_hi},
            {"$K_2$ (km/s)",     dq.K2,      dq.K2_err,
             ob.has_K2, ob.K2_obs, ob.K2_err_lo, ob.K2_err_hi},
            {"$R_1$ ($R_\\odot$)",  dq.R1,   dq.R1_err,
             ob.has_R1, ob.R1_obs, ob.R1_err_lo, ob.R1_err_hi},
            {"$R_2$ ($R_\\odot$)",  dq.R2,   dq.R2_err,
             ob.has_R2, ob.R2_obs, ob.R2_err_lo, ob.R2_err_hi},
            {"$M_1$ ($M_\\odot$)",  dq.M1,   dq.M1_err,
             ob.has_M1, ob.M1_obs, ob.M1_err_lo, ob.M1_err_hi},
            {"$M_2$ ($M_\\odot$)",  dq.M2,   dq.M2_err,
             ob.has_M2, ob.M2_obs, ob.M2_err_lo, ob.M2_err_hi},
            {"$M_{\\rm total}$ ($M_\\odot$)", dq.M_total, dq.M_total_err,
             ob.has_Mtotal, ob.Mtotal_obs, ob.Mtotal_err_lo, ob.Mtotal_err_hi},
            {"$q$",              dq.q,       dq.q_err,
             ob.has_q,  ob.q_obs,  ob.q_err_lo,  ob.q_err_hi},
            {"$\\log g_1$ (dex)", dq.logg1,  dq.logg1_err,
             ob.has_logg1, ob.logg1_obs, ob.logg1_err_lo, ob.logg1_err_hi},
            {"$\\log g_2$ (dex)", dq.logg2,  dq.logg2_err,
             ob.has_logg2, ob.logg2_obs, ob.logg2_err_lo, ob.logg2_err_hi},
            {"$T_1$ (K)",        dq.t1,      dq.t1_err,
             ob.has_T1, ob.T1_obs, ob.T1_err_lo, ob.T1_err_hi},
            {"$T_2$ (K)",        dq.t2,      dq.t2_err,
             ob.has_T2, ob.T2_obs, ob.T2_err_lo, ob.T2_err_hi},
            {"$a$ ($R_\\odot$)", dq.a_rsun,  dq.a_rsun_err,
             false, 0, 0, 0},
        };

        for (auto& r : rows) {
            if (r.impl == 0.0 && r.impl_err == 0.0 && !r.has_obs)
                continue;
            out << r.label << " & ";
            if (dq.has_errors && r.impl_err > 0)
                out << tex_val_pm(r.impl, r.impl_err);
            else {
                std::ostringstream o;
                o << std::fixed << std::setprecision(4) << "$" << r.impl << "$";
                out << o.str();
            }
            out << " & ";
            if (r.has_obs)
                out << tex_obs(r.obs_v, r.obs_lo, r.obs_hi);
            else
                out << "{\\color{gray}---}";
            out << " & ";
            if (r.has_obs) {
                double pull = compute_pull(r.impl, r.obs_v, r.obs_lo, r.obs_hi);
                std::ostringstream o;
                o << std::fixed << std::setprecision(2) << pull;
                std::string col = (std::abs(pull) < 2.0) ? "green!70!black"
                                : (std::abs(pull) < 3.0) ? "orange!80!black"
                                : "red!80!black";
                out << "\\textbf{\\textcolor{" << col << "}{$"
                    << o.str() << "\\sigma$}}";
            } else {
                out << "{\\color{gray}---}";
            }
            out << " \\\\\n";
        }
        out << "\\bottomrule\n"
            << "\\end{tabular}\n"
            << "\\rowcolors{0}{}{}\n\n";
    }

    // ═══════════════ Section 4: Prior Residuals ══════════════════════
    if (rd.has_priors && !rd.prior_names.empty()) {
        out << "\\section{Prior Constraint Residuals}\n\n"
            << "{\\small Residuals include prior weighting and balance scaling.}\n"
            << "\\vspace{0.5em}\n\n"
            << "\\rowcolors{2}{rowA}{rowB}\n"
            << "\\begin{tabular}{l r l}\n"
            << "\\rowcolor{headerblue}\n"
            << "\\thead{Constraint} & \\thead{Scaled Residual} & \\thead{Status} \\\\\n"
            << "\\toprule\n";
        for (size_t k = 0; k < rd.prior_names.size(); ++k) {
            double r = (k < rd.prior_residuals.size())
                     ? rd.prior_residuals[k] : 0.0;
            std::string status, col, icon;
            if (std::abs(r) < 2.0) {
                status = "OK";
                col    = "green!70!black";
                icon   = "$\\checkmark$";
            } else if (std::abs(r) < 3.0) {
                status = "Tension";
                col    = "orange!80!black";
                icon   = "$\\sim$";
            } else {
                status = "Discrepant";
                col    = "red!80!black";
                icon   = "$\\boldsymbol{\\times}$";
            }
            std::ostringstream o;
            o << std::fixed << std::setprecision(3) << r;
            out << tex_escape(rd.prior_names[k]) << " & $"
                << o.str() << "\\sigma$ & \\textbf{\\textcolor{" << col << "}{"
                << icon << " " << status << "}} \\\\\n";
        }
        out << "\\bottomrule\n"
            << "\\end{tabular}\n"
            << "\\rowcolors{0}{}{}\n\n";
    }

    // ═══════════════ Section 5: Correlation Matrix ═══════════════════
    if (rd.has_covariance && rd.npar <= 15
        && (int)rd.correlations.size() == rd.npar)
    {
        out << "\\section{Correlation Matrix}\n\n";

        // Use footnotesize for larger matrices
        if (rd.npar > 8)
            out << "{\\footnotesize\n";
        else
            out << "{\\small\n";

        out << "\\setlength{\\tabcolsep}{4pt}\n"
            << "\\begin{tabular}{l" << std::string(rd.npar, 'r') << "}\n"
            << "\\toprule\n";
        out << "\\rowcolor{headerblue} ";
        out << "\\thead{}";
        for (int j = 0; j < rd.npar; ++j)
            out << " & \\thead{" << param_to_latex_short(rd.par_names[j]) << "}";
        out << " \\\\\n\\midrule\n";

        for (int i = 0; i < rd.npar; ++i) {
            if (i % 2 == 0)
                out << "\\rowcolor{rowA}";
            else
                out << "\\rowcolor{rowB}";
            out << param_to_latex_short(rd.par_names[i]);
            for (int j = 0; j < rd.npar; ++j) {
                double r = rd.correlations[i][j];
                out << " & ";
                if (i == j) {
                    out << "\\cellcolor{accentblue!15}$\\mathbf{1}$";
                } else {
                    std::string bg;
                    if (std::abs(r) > .70)      bg = "\\cellcolor{red!20}";
                    else if (std::abs(r) > .40)  bg = "\\cellcolor{yellow!20}";

                    std::ostringstream o;
                    // Show only magnitude off-diagonal, sign via color
                    o << std::fixed << std::setprecision(2) << r;
                    std::string numcol = (std::abs(r) > 0.70) ? "red!70!black"
                                       : (std::abs(r) > 0.40) ? "orange!80!black"
                                       : "black";
                    out << bg << "{\\color{" << numcol << "}$" << o.str() << "$}";
                }
            }
            out << " \\\\\n";
        }
        out << "\\bottomrule\n"
            << "\\end{tabular}\n}\n\n";
    }

    // ── footer ──
    out << "\\vfill\n"
        << "\\begin{center}\n"
        << "{\\small\\color{gray} Generated by \\textsc{LCURVE} --- "
        << "\\today}\n"
        << "\\end{center}\n\n";

    out << "\\end{document}\n";
    out.close();
    std::cout << "TeX report written to " << tex_path << std::endl;
}

// ─────────────────────────────────────────────────────────────────────
inline void try_compile_pdf(const std::string& tex_path)
{
    size_t slash = tex_path.rfind('/');
    std::string dir   = (slash != std::string::npos)
                      ? tex_path.substr(0, slash) : ".";
    std::string fname = (slash != std::string::npos)
                      ? tex_path.substr(slash + 1) : tex_path;

    std::string cmd = "cd \"" + dir + "\" && pdflatex -interaction=nonstopmode \""
                    + fname + "\" > /dev/null 2>&1";
    int ret = std::system(cmd.c_str());
    if (ret == 0) {
        std::system(cmd.c_str());

        std::string base = fname;
        auto dot = base.rfind('.');
        if (dot != std::string::npos) base = base.substr(0, dot);
        std::string cleanup = "cd \"" + dir + "\" && rm -f \""
            + base + ".aux\" \"" + base + ".log\" \""
            + base + ".out\" 2>/dev/null";
        std::system(cleanup.c_str());

        std::string pdf = dir + "/" + base + ".pdf";
        std::cout << "PDF report compiled: " << pdf << std::endl;
    }
}

} // namespace ReportWriter