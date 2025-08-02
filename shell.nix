let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (python-pkgs: [
      python-pkgs.pandas
      python-pkgs.requests
      python-pkgs.openai
      python-pkgs.pydantic
      python-pkgs.bpython
      python-pkgs.litellm
      python-pkgs.matplotlib
      python-pkgs.networkx
      python-pkgs.pygraphviz
      python-pkgs.pytest
      python-pkgs.rich
      python-pkgs.mypy
      python-pkgs.colorama
      python-pkgs.google-api-python-client 
      python-pkgs.google-auth-httplib2 
      python-pkgs.google-auth-oauthlib
    ]))
  ];
}
