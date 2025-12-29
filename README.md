# resample-extremes

Python script to resample multi-resolution timeseries containing extreme periods of high frequency burst data.

![image](https://github.com/patternizer/resample-extremes/blob/main/OUT/v1/raw_vs_monthly_naive_vs_censored.png)

## Create environment

	conda create -n resample-extremes python=3.11 -y
	conda activate resample-extremes

## Install dependencies

	pip install -r requirements.txt

## Run analysis

	  python resample-extremes.py --outdir OUT --seed 0 --progress
	  python resample-extremes.py --outdir OUT --seed 0 --progress --n-boot 200 --ceemdan-trials 10
	  python resample-extremes.py --outdir OUT --seed 0 --progress --n-boot 200 --ceemdan-trials 10 --skip-wavelet

## 🤝 Contributing

Issues and PRs are welcome. When filing an issue, please include:

- a minimal CSV sample (or a snippet),
- the exact command you ran,
- the observed vs. expected output (and a PNG if possible).

 Contact information:

* [Michael Taylor](https://patternizer.github.io/)

---

## 📄 License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
