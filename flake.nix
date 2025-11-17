{
  description = "ShapingRL - Reward Shaping and Potential-Based Methods in RL";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs, ... }:
    let
      # Disable ROCm documentation build to avoid LaTeX dependency issues
      rocmOverlay = final: prev: {
        rocmPackages = prev.rocmPackages.overrideScope (
          final': prev': {
            rocdbgapi = prev'.rocdbgapi.overrideAttrs (old: {
              outputs = [ "out" ];
              cmakeFlags = (old.cmakeFlags or [ ]) ++ [ "-DBUILD_DOC=OFF" ];
              postBuild = "true";
              postInstall = "true";
            });
          }
        );
      };

      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
        overlays = [ rocmOverlay ];
      };

      python = pkgs.python313;

      # Use torchWithRocm from nixpkgs (ROCm 6.4.3, PyTorch 2.9.0)
      # This is the clean nixpkgs approach - if available in cache, no compilation needed
      # ROCm 6.4.3 should support gfx1201 with HSA_OVERRIDE_GFX_VERSION workaround
      pythonEnv = python.withPackages (
        ps: with ps; [
          torchWithRocm # Deep learning framework for DQN and PPO agents (src/agents/dqn.py, src/agents/ppo.py)
          gymnasium # RL environments: MountainCar and Atari (src/environments/mountain_car.py, src/environments/montezumas_revenge.py)
          ale-py # Atari Learning Environment for Montezuma's Revenge (src/environments/montezumas_revenge.py)
          pygame # Environment rendering and display (used by gymnasium for visual environments)
          numpy # Scientific computing, used throughout codebase for array operations
          matplotlib # Visualization in notebooks (notebooks/exploratory_analysis.ipynb)
          jupyter # Interactive notebooks for experiments and analysis
          pytest # Testing framework (tests/ directory)
          ruff # Code formatting and linting (used in development workflow)
          opencv4 # Image processing for Montezuma's Revenge test wrapper (tests/information_extraction/)
          wandb # Experiment tracking and logging (src/core/wandb_callback.py, used in training scripts)
          imageio # Video recording for wandb (enables saving environment videos to wandb)
          imageio-ffmpeg # FFmpeg backend for imageio video encoding
          moviepy # Video editing/processing for wandb video recording
          pillow # Image processing library, dependency for wandb video recording
          (stable-baselines3.override {
            # RL algorithm implementations (PPO, DQN wrappers in src/agents/)
            torch = torchWithRocm; # Use ROCm-enabled PyTorch for GPU support
          })
        ]
      );
    in
    {
      devShells.x86_64-linux.default = pkgs.mkShell {
        packages = with pkgs; [
          curl # Download ALE ROMs in shellHook (setup_ale_roms function)
          unrar # Extract ALE ROMs archive in shellHook
          sdl3 # SDL3 library required by ALE/gymnasium for environment rendering
          sdl3-image # SDL3 image support for ALE environments
          sdl3-ttf # SDL3 font rendering for ALE environments
          libGL # OpenGL library for GPU-accelerated rendering in environments
          libGLU # OpenGL Utility library, dependency for OpenGL rendering
          xorg.libX11 # X11 core library, required by SDL/OpenGL for windowing
          xorg.libXext # X11 extensions, required by SDL for extended functionality
          xorg.libXrandr # X11 RandR extension for display configuration
          xorg.libXinerama # X11 Xinerama extension for multi-monitor support
          xorg.libXi # X11 Input extension for input handling in environments
          xorg.libXcursor # X11 cursor library for mouse cursor support
          pythonEnv # Python environment with all RL dependencies
          rocmPackages.rocm-runtime # ROCm runtime libraries for AMD GPU support with PyTorch
        ];

        shellHook = ''
          # ALE ROMs directory (lazy setup via setup_ale_roms function)
          export ALE_ROMS_DIR="$HOME/.ale_roms"

          # SDL environment variables for ALE/gymnasium rendering
          export SDL_VIDEODRIVER=x11
          export SDL_AUDIODRIVER=pulse
          export LD_LIBRARY_PATH="${pkgs.sdl3}/lib:${pkgs.sdl3-image}/lib:${pkgs.sdl3-ttf}/lib:\$LD_LIBRARY_PATH"

          # GPU configuration for Wayland/Hyprland
          # System-wide settings from /etc/nixos/modules: RADV_PERFTEST=gpl, AMD_VULKAN_ICD=RADV,
          # WLR_DRM_NO_MODIFIERS=1, MESA_VK_DEVICE_SELECT=0 (enable GPU sharing)
          if [ -n "$WAYLAND_DISPLAY" ] || [ -n "$HYPRLAND_INSTANCE_SIGNATURE" ]; then
            # Project-specific workarounds to prevent Hyprland crashes
            export WLR_DRM_NO_ATOMIC=1
            export WLR_NO_HARDWARE_CURSORS=1
            export HIP_VISIBLE_DEVICES=0
            export PYTORCH_ALLOC_CONF=max_split_size_mb:128
          fi

          # PyTorch GPU memory management fallback for non-Wayland
          if [ -z "$PYTORCH_ALLOC_CONF" ]; then
            export PYTORCH_ALLOC_CONF=max_split_size_mb:128
          fi

          # Lazy ROM download (only runs when setup_ale_roms is called)
          setup_ale_roms() {
            if [ ! -d "$HOME/.ale_roms" ]; then
              echo "Downloading ALE ROMs..."
              mkdir -p "$HOME/.ale_roms"
              cd "$HOME/.ale_roms"
              ${pkgs.curl}/bin/curl -L "http://www.atarimania.com/roms/Roms.rar" -o roms.rar
              ${pkgs.unrar}/bin/unrar x -o+ roms.rar .
              find . -mindepth 2 -name "*.bin" -exec mv {} . \;
              find . -type d -empty -delete
              if [ -f "Montezuma's Revenge - Featuring Panama Joe.bin" ]; then
                ln -sf "Montezuma's Revenge - Featuring Panama Joe.bin" montezuma_revenge.bin
              fi
              rm -f roms.rar
              echo "ALE ROMs setup complete!"
            fi
          }

          # Register Jupyter kernel for Cursor auto-selection (lazy - only runs when called)
          setup_jupyter_kernel() {
            if ! ${pythonEnv}/bin/python -m ipykernel list 2>/dev/null | grep -q "RL"; then
              ${pythonEnv}/bin/python -m ipykernel install --user --name RL --display-name "RL" >/dev/null 2>&1 || true
            fi
          }
        '';
      };
    };
}
