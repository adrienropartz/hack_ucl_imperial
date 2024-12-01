{
  description = "Python venv development env in nix";

  inputs = {
    utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    utils,
    ...
  }:
    utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
      pythonPackages = pkgs.python311Packages;
    in {
      devShells.default = pkgs.mkShell {
        name = "python-venv";
        venvDir = "./.venv";
        buildInputs = [
          pkgs.conda
        ];
        # Run this command, only after creating the virtual environment
        postVenvCreation = ''
          unset SOURCE_DATE_EPOCH
          pip install --upgrade pip
          pip install -r requirements.txt
        '';

        # Now we can execute any commands within the virtual environment.
        # This is optional and can be left out to run pip manually.
        postShellHook = ''
          # allow pip to install wheels
          unset SOURCE_DATE_EPOCH
        '';

        # Dependency for NixOS
        LD_LIBRARY_PATH = ''${pkgs.stdenv.cc.cc.lib}/lib/:${pkgs.libGL}/lib/:${pkgs.glib.out}/lib:/run/opengl-driver/lib/:${pkgs.zlib}/lib'';
      };
    });
}
