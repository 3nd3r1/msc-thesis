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
in
pkgs.mkShell {
  buildInputs = with pkgs; [
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
    python3Packages.geopandas
    python3Packages.transformers
    python3Packages.torch
    python3Packages.rich
    lotus-ai
  ];
}
