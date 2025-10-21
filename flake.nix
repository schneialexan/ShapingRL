{
  description = "ShapingRL - Reward Shaping and Potential-Based Methods in RL";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs, ... }:
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

          # 1. Get the Python version we will be working with.
          python = pkgs.python313;

          # 2. Use standard ale-py and handle ROMs in shellHook
          ale-py-with-roms = pkgs.python3.pkgs.ale-py;
          

        in
        {
          # Default: ROCm (Linux) or Metal (macOS)
          default = pkgs.mkShell {
            packages = with pkgs; [
              git
              jujutsu
              lazyjj
              curl
              unrar
              # SDL3 for ale-py compatibility
              sdl3
              sdl3-image
              sdl3-ttf
              # Additional libraries for ale-py compatibility
              libGL
              libGLU
              xorg.libX11
              xorg.libXext
              xorg.libXrandr
              xorg.libXinerama
              xorg.libXi
              xorg.libXcursor
              # 5. Use our custom python environment with the new package.
              (python.withPackages (ps: with ps; [
                (if stdenv.isLinux then torchWithRocm else torch)
                gymnasium
                ale-py-with-roms # <-- Use our custom, self-contained package
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
                wandb
                (stable-baselines3.override {
                  torch = if stdenv.isLinux then torchWithRocm else torch;
                })
              ]))
            ] ++ pkgs.lib.optionals stdenv.isLinux [
              rocmPackages.rocm-runtime
              rocmPackages.rocminfo
            ];

            # 6. Setup ALE ROMs and GPU environment
            shellHook = ''
              # Download and setup ALE ROMs if not already present
              if [ ! -d "$HOME/.ale_roms" ]; then
                echo "Downloading ALE ROMs..."
                mkdir -p "$HOME/.ale_roms"
                cd "$HOME/.ale_roms"
                ${pkgs.curl}/bin/curl -L "http://www.atarimania.com/roms/Roms.rar" -o roms.rar
                ${pkgs.unrar}/bin/unrar x -o+ roms.rar .
                # Move ROMs from subdirectories to main directory
                find . -mindepth 2 -name "*.bin" -exec mv {} . \;
                # Clean up empty subdirectories
                find . -type d -empty -delete
                # Create symlinks for expected ROM names
                if [ -f "Montezuma's Revenge - Featuring Panama Joe.bin" ]; then
                  ln -sf "Montezuma's Revenge - Featuring Panama Joe.bin" montezuma_revenge.bin
                fi
                rm -f roms.rar
                echo "ALE ROMs setup complete!"
              fi
              export ALE_ROMS_DIR="$HOME/.ale_roms"
              
              # Set SDL environment variables for better compatibility
              export SDL_VIDEODRIVER=x11
              export SDL_AUDIODRIVER=pulse
              export SDL_VIDEO_WINDOW_POS=0,0
              
              # Help ale-py find SDL3 libraries
              export LD_LIBRARY_PATH="${pkgs.sdl3}/lib:${pkgs.sdl3-image}/lib:${pkgs.sdl3-ttf}/lib:\$LD_LIBRARY_PATH"
              export PKG_CONFIG_PATH="${pkgs.sdl3}/lib/pkgconfig:${pkgs.sdl3-image}/lib/pkgconfig:${pkgs.sdl3-ttf}/lib/pkgconfig:\$PKG_CONFIG_PATH"
              
              ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
                export HSA_OVERRIDE_GFX_VERSION=11.0.1
                export ROCR_VISIBLE_DEVICES=0
                if [ -e /dev/nvidia0 ]; then
                  echo "⚠️  NVIDIA GPU detected! Use: nix develop .#cuda"
                fi
              ''}
            '';
          };

          # NVIDIA: Explicit CUDA support
          cuda = pkgs.mkShell {
            packages = with pkgs; [
              git
              jujutsu
              lazyjj
              curl
              unrar
              # SDL3 for ale-py compatibility
              sdl3
              sdl3-image
              sdl3-ttf
              # Additional libraries for ale-py compatibility
              libGL
              libGLU
              xorg.libX11
              xorg.libXext
              xorg.libXrandr
              xorg.libXinerama
              xorg.libXi
              xorg.libXcursor
              (python.withPackages (ps: with ps; [
                torchWithCuda
                gymnasium
                ale-py-with-roms # <-- Also use our custom package here
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
                wandb
                (stable-baselines3.override {
                  torch = if stdenv.isLinux then torchWithRocm else torch;
                })
              ]))
              cudaPackages.cudatoolkit
            ];

            # Setup ALE ROMs and CUDA environment
            shellHook = ''
              # Download and setup ALE ROMs if not already present
              if [ ! -d "$HOME/.ale_roms" ]; then
                echo "Downloading ALE ROMs..."
                mkdir -p "$HOME/.ale_roms"
                cd "$HOME/.ale_roms"
                ${pkgs.curl}/bin/curl -L "http://www.atarimania.com/roms/Roms.rar" -o roms.rar
                ${pkgs.unrar}/bin/unrar x -o+ roms.rar .
                # Move ROMs from subdirectories to main directory
                find . -mindepth 2 -name "*.bin" -exec mv {} . \;
                # Clean up empty subdirectories
                find . -type d -empty -delete
                # Create symlinks for expected ROM names
                if [ -f "Montezuma's Revenge - Featuring Panama Joe.bin" ]; then
                  ln -sf "Montezuma's Revenge - Featuring Panama Joe.bin" montezuma_revenge.bin
                fi
                rm -f roms.rar
                echo "ALE ROMs setup complete!"
              fi
              export ALE_ROMS_DIR="$HOME/.ale_roms"
              
              # Set SDL environment variables for better compatibility
              export SDL_VIDEODRIVER=x11
              export SDL_AUDIODRIVER=pulse
              export SDL_VIDEO_WINDOW_POS=0,0
              
              # Help ale-py find SDL3 libraries
              export LD_LIBRARY_PATH="${pkgs.sdl3}/lib:${pkgs.sdl3-image}/lib:${pkgs.sdl3-ttf}/lib:\$LD_LIBRARY_PATH"
              export PKG_CONFIG_PATH="${pkgs.sdl3}/lib/pkgconfig:${pkgs.sdl3-image}/lib/pkgconfig:${pkgs.sdl3-ttf}/lib/pkgconfig:\$PKG_CONFIG_PATH"
              
              export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
            '';
          };
        });
    };
}