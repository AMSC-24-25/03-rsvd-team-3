# Project setup
For `macos`, install `CMake` and `Vcpkg`:
```bash
brew install cmake
brew install vcpkg
```
Add to `PATH`:
!**Change `{{your username}}` to your own username**!
```bash
vim ~/.zshrc

# Add
export VCPKG_ROOT=/Users/{{your username}}/vcpkg
export PATH=$VCPKG_ROOT:$PATH
```

For Windows, see this guide: https://learn.microsoft.com/en-us/vcpkg/get_started/get-started/

**Fork** and **Clone** this project to your own repo.

And then write building script, add to project root directory:
## CMakeUserPresets.json
```json
{
  "version": 2,
  "configurePresets": [
    {
      "name": "default",
      "inherits": "vcpkg",
      "environment": {
        "VCPKG_ROOT": "xxxxxxxxxxxxxxxxxxxx/vcpkg"
      },
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      }
    }
  ]
}

```

Finally, configure the build using CMake:

1.  Configure the build using CMake:
```
cmake --preset=default
```
2. Build the project
```
cmake --build build
```
3. Run the application
```
./build/main
```
### or
```
mpirun -np 4 ./build/main
```