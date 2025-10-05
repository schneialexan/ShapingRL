{
  description = "ShapingRL - Reward Shaping and Potential-Based Methods in RL";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
        in
        {
          # Default: ROCm (Linux) or Metal (macOS)
          default = pkgs.mkShell {
            packages = with pkgs; [
              git
              jujutsu
              lazyjj
              (python313.withPackages (ps: with ps; [
                (if stdenv.isLinux then torchWithRocm else torch)
                gymnasium
                ale-py
                pygame
                numpy
                pandas
                scipy
                matplotlib
                seaborn
                jupyter
                pytest
                black
                flake8
                tqdm
                pyyaml
              ]))
            ] ++ pkgs.lib.optionals stdenv.isLinux [
              rocmPackages.rocm-runtime
              rocmPackages.rocminfo
            ];

            shellHook = pkgs.lib.optionalString pkgs.stdenv.isLinux ''
              export HSA_OVERRIDE_GFX_VERSION=11.0.1
              export ROCR_VISIBLE_DEVICES=0
              if [ -e /dev/nvidia0 ]; then
                echo "⚠️  NVIDIA GPU detected! Use: nix develop .#cuda"
              fi
            '';
          };

          # NVIDIA: Explicit CUDA support
          cuda = pkgs.mkShell {
            packages = with pkgs; [
              git
              jujutsu
              lazyjj
              (python313.withPackages (ps: with ps; [
                torchWithCuda
                gymnasium
                ale-py
                pygame
                numpy
                pandas
                scipy
                matplotlib
                seaborn
                jupyter
                pytest
                black
                flake8
                tqdm
                pyyaml
              ]))
              cudaPackages.cudatoolkit
            ];

            shellHook = ''
              export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
            '';
          };
        });
    };
}