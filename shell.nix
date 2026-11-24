{
  pkgs ? import <nixpkgs> { },
}:

let
  ps = pkgs.python3Packages;

  lotus-ai = ps.buildPythonPackage rec {
    pname = "lotus-ai";
    version = "1.2.4";
    pyproject = true;

    src = ps.fetchPypi {
      pname = "lotus_ai";
      inherit version;
      hash = "sha256-A/EWrUry/qIGYRfO1lk4qWxMyB7WFjFc8Y6GKymmVH0=";
    };

    build-system = [ ps.hatchling ];

    pythonRelaxDeps = [
      "numpy"
      "pandas"
      "sentence-transformers"
    ];
    pythonRemoveDeps = [ "faiss-cpu" ];

    dependencies = with ps; [
      backoff
      faiss
      litellm
      numpy
      pandas
      pillow
      pydantic
      requests
      sentence-transformers
      tiktoken
      tqdm
    ];

    pythonImportsCheck = [ "lotus" ];
  };

  bargain = ps.buildPythonPackage rec {
    pname = "bargain";
    version = "0.1.3";
    pyproject = true;

    src = ps.fetchPypi {
      inherit pname version;
      hash = "sha256-dgUW4wTVAWEdBK0Ypagyg1muWg24ZbAOhB9AiYjghP4=";
    };

    build-system = [ ps.setuptools ];

    dependencies = with ps; [
      numpy
      openai
      pandas
      tqdm
    ];

    pythonImportsCheck = [ "BARGAIN" ];
  };
  verdaVersion = "1.10.0"; # github.com/verda-cloud/verda-cli, not in nixpkgs
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    go
    sqlite
    python3
    python3Packages.pip
    python3Packages.matplotlib
    python3Packages.numpy
    python3Packages.pandas
    python3Packages.networkx
    python3Packages.seaborn
    python3Packages.scipy
    python3Packages.scikit-learn
    python3Packages.nltk
    python3Packages.wordcloud
    python3Packages.jupytext
    python3Packages.jupyterlab
    python3Packages.transformers
    python3Packages.torch
    python3Packages.torch-geometric
    python3Packages.rich
    lotus-ai
    bargain
  ];
  shellHook = ''
    verdaBin="$PWD/.bin/verda-${verdaVersion}"
    export PATH="$verdaBin:$PATH"
    if [ ! -x "$verdaBin/verda" ]; then
      echo "Installing verda-cli ${verdaVersion}..."
      GOBIN="$verdaBin" GOFLAGS=-modcacherw go install github.com/verda-cloud/verda-cli/cmd/verda@v${verdaVersion}
    fi
  '';
}
