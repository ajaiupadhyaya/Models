"""
Setup script for financial models workspace.
"""

import subprocess
import sys
import os


def check_python_version():
    """Check if Python version is 3.8+."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_requirements():
    """Install required packages."""
    print("\nInstalling requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Error installing requirements")
        sys.exit(1)


def create_directories():
    """Create necessary directories."""
    directories = [
        'data',
        'notebooks',
        'outputs',
        'reports',
        'presentations',
        'config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def create_env_file():
    """Create .env file if it doesn't exist."""
    env_path = '.env'
    env_example_path = '.env.example'
    
    if not os.path.exists(env_path):
        if os.path.exists(env_example_path):
            with open(env_example_path, 'r') as f:
                content = f.read()
            with open(env_path, 'w') as f:
                f.write(content)
            print("✓ Created .env file from template")
            print("  Please update .env with your API keys")
        else:
            print("⚠ .env.example not found, skipping .env creation")
    else:
        print("✓ .env file already exists")


def main():
    """Run setup."""
    print("=" * 50)
    print("Financial Models Workspace Setup")
    print("=" * 50)
    
    check_python_version()
    create_directories()
    create_env_file()
    
    install_choice = input("\nInstall requirements? (y/n): ").lower()
    if install_choice == 'y':
        install_requirements()
    else:
        print("Skipping requirement installation")
        print("Run: pip install -r requirements.txt")
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Update .env file with your API keys")
    print("2. Run: jupyter lab")
    print("3. Open notebooks/ for examples")


if __name__ == '__main__':
    main()
