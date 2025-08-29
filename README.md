# lcurve_re

Minimal, modern re-adaptation of Tom Marsh’s **lcurve** code.
The goal is to make binary-star light-curve modelling easy to build,
easy to script and easy to extend on any recent Linux / macOS box.

* Stand-alone C++17/20 implementation – no ancient Fortran, no IRAF
* Fast static‐geometry engine with OpenMP parallelism
* Optional mass–ratio / velocity-scale / radius-scale Bayesian priors
* Command-line tools for direct evaluation, Nelder–Mead optimisation and
  full MCMC sampling
* Interactive, live-updating plots via **gnuplot-iostream**

---

## 1 Quick start

```bash
# 1. Clone
git clone https://github.com/yourname/lcurve_re.git
cd lcurve_re

# 2. Configure & build (uses CMake ≥ 3.14)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# 3. Run one of the tools
./build/lcurve_re          path/to/config.json      # single light-curve evaluation
./build/mcmc_solver        path/to/config.json      # Metropolis-Hastings / MCMC
./build/simplex            path/to/config.json      # Nelder–Mead optimiser
```

All executables understand exactly one argument – the
JSON configuration file described below.

---

## 2 Binary overview

| Executable                       | Purpose                                                               |
|---------------------------------|-----------------------------------------------------------------------|
| `lcurve_re` (main.cpp)          | Evaluate a model, scale it to the data and write the best-fit curve   |
| `mcmc_solver`                   | Metropolis–Hastings sampler with live χ² plot & colourful progress bar|
| `simplex`                       | Lightweight Nelder–Mead optimiser (no priors, no chains)              |

All binaries live in `build/` after compilation.

---

## 3 Configuration file (abridged)

```jsonc
{
  /* --- required --------------------------------------------------- */
  "data_file_path"    : "lightcurve.dat",        // or "none" for synthetic data
  "output_file_path"  : "fit_out.dat",

  /* --- global switches -------------------------------------------- */
  "autoscale"         : true,        // let the code re-scale the model to χ² min
  "plot_device"       : "qt",        // qt | wxt | x11 | pngcairo | none
  "seed"              : -123456,     // RNG seed  (negative = random)

  /* --- noise / fake data block ------------------------------------ */
  "noise"             : 1.5e-3,      // Gaussian σ added to ‘data’ for writing

  /* --- MCMC -------------------------------------------------------- */
  "mcmc_steps"        : 40000,
  "mcmc_burn_in"      : 10000,
  "use_priors"        : true,
  "priors" : {
      "vrad1_obs" : "186.2  2.0   0",   // mean ±1σ  [km s⁻¹]
      "m1"        : "0.82   0.17  0",   // M☉
      "m2_min"    : "1.30   0.27  0",   // M☉
      "r1"        : "0.309  0.020 0"    // R☉
  },

  /* --- model ------------------------------------------------------- */
  "model_parameters" : {
      "q"          : "1.4   0.5  0.01  1 1",   // value range dstep vary defined
      "iangle"     : "65.0  5    0.1   1 1",
      "r1"         : "0.014 0.01 0.0001 0 1",
      "... many more see src/model.cpp ..."
  }
}
```

•  The five numbers in each `model_parameters` entry are  
   `value  range  dstep  vary(0|1)  defined(0|1)` – identical to the
   original *lcurve*  files.  
•  If `autoscale : true` a single scale factor (or four individual
   factors when `iscale=true` in the model) is fitted to minimise χ².  
•  `plot_device : "none"` turns off all gnuplot calls – useful for
   headless clusters.

---

## 4 Directory structure

```
lcurve_re/
│
├─ new_scripts/       » small command-line front-ends
│   ├─ mcmc_solver.cpp
│   ├─ simplex.cpp
│   └─ test.cpp
│
├─ src/               » reusable library
│   ├─ lcurve_base/   – light-curve engine (modernised Tom Marsh code)
│   ├─ lroche_base/   – Roche geometry, stream & eclipse maths
│   ├─ mass_ratio_pdf – fast 2-D KDE grid for inclination-conditioned priors
│   └─ new_subs.*     – lightweight replacement of TRM’s “subs” helpers
│
├─ main.cpp           » minimal evaluator
└─ CMakeLists.txt     » build recipe
```

---

## 5 Dependencies

Run-time  
• gnuplot (≥ 5.0) – only when `plot_device` ≠ "none"

Build-time  
• CMake ≥ 3.14  
• A C++17 (or later) compiler (gcc-10+, clang-11+, MSVC 2019)  
• `nlohmann/json` (header-only, bundled as a git submodule)  
• `gnuplot-iostream` (header-only, bundled)  
• Optional: OpenMP for multi-threaded geometry / KDE

All third-party headers are vendored – nothing is fetched at build time.

---

## 6 Typical workflow

1.  Prepare a first-guess configuration  
    `cp examples/template.json myrun.json`  

2.  Optimise with Nelder–Mead  
    ```bash
    ./build/simplex  myrun.json
    ```

3.  Run a full MCMC with priors  
    ```bash
    ./build/mcmc_solver  myrun.json
    ```

4.  Inspect `chain_out.txt` in your favourite corner-plotter, tweak
    steps / priors and repeat.

---

## 9 License

Original algorithms © Tom Marsh, 2001-2023.  
Re-written C++17 adaption, helpers and build scripts © 2025 Fabian
Mattig, MIT licence.  See `LICENSE` files for full text.

---

Happy modelling! – *Fabian*
