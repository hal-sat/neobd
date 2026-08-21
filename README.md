# neobd

Pure Python tools for preprocessing microtremor array records and running
SPAC, GSPAC, FK, and DSPAC analyses.

## Install and run

```console
pip install .
neobd path/to/params.json
```

Inputs use the original `params.json`, `array_coord.csv`, receiver CSV, and
optional `valid_segments.csv` layouts. Receiver records may be single files or
CSV blocks in receiver directories. The sampling interval is the median of all
intervals in the input. Results are written below `results_neobd/`.

Set the shared process count at the top level:

```json
"n_para": 4
```

Override it for one run with `--npara`; `0` uses all available CPUs. If neither
the CLI option nor `n_para` is specified, one process is used.

```console
neobd path/to/params.json --npara=4
neobd path/to/params.json --npara=0
```

Choose frequency-domain smoothing with either a Parzen bandwidth in hertz or
the number of repeated three-point Hann passes:

```json
"smoothing": {"type": "Parzen", "params": [0.3]}
```

```json
"smoothing": {"type": "Hann_3point", "params": [7]}
```

The legacy `"n_smoothing": 7` form remains equivalent to `Hann_3point`.

## Analysis settings

```json
{
  "SPAC": {
    "01p5": ["R04", "R01", "R04", "R02", "R04", "R03"]
  },
  "GSPAC": {
    "01p5": {
      "center": "R04",
      "ring": ["R01", "R02", "R03"],
      "methods": ["cca", "v", "h0", "h1"]
    }
  },
  "FK": {
    "methods": ["mlm", "bfm"],
    "diagonal_loading": 0.02,
    "output_interval": 10,
    "max_frequency": 15,
    "density": [200, 36],
    "bounds": [100, 1500],
    "optimizer": {
      "method": "de",
      "population": 80,
      "iterations": 120,
      "seed": 1213
    }
  },
  "DSPAC": {
    "array": ["R02", "R03", "R04"],
    "max_frequency": 5,
    "optimizer": {
      "method": "pso",
      "population": 1000,
      "iterations": 1000,
      "target": 1e-10,
      "patience": 20,
      "inertia": [0.9, 0.4],
      "w4loc": 1.4,
      "w4glo": 0.7,
      "seed": 1
    }
  }
}
```

FK and DSPAC independently support `"de"` and `"pso"`. DE uses `mutation`
and `crossover`; PSO uses `inertia`, `w4loc`, and `w4glo`. Normally omit
`target` for FK because its minimized objective is negative.

FK outputs are separated into `results_neobd/fk/MLM/` and
`results_neobd/fk/BFM/`. GSPAC outputs are grouped under
`results_neobd/gspac/<method>/`.

CCA uses only ring receivers and estimates the circle center from their
coordinates. A `center` receiver is required only for V, H0, and H1. Circular
spectra are written below `results_neobd/circular_statistics/<array>/`.

## FK visualization

```console
neobd visualize-fk results_neobd/fk/MLM/FK_04p44410_Hz.csv
```

The command opens Matplotlib by default. Use `--output map.png` to save the
figure, `--no-show` to suppress the window, or `--db` for decibels.

## Development

```console
pip install -e '.[test]'
pytest
black src tests
```
