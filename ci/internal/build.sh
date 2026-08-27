#!/usr/bin/env bash
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Build hipCollections: the tests, benchmarks and examples, the DEB/RPM
# packages and the tests tarball that the tester images consume.
#
# Required environment variables:
#   SRC_DIR              Parent directory of the project checkout.
#   BUILD_DIR            Scratch directory for the build tree.
#   INSTALL_DIR          Prefix the install step targets and the packages are
#                        built for.
#   BUILD_ARTIFACTS_DIR  Destination for packages and the tests tarball.
#
# Optional environment variables:
#   PROJECT_ID                       Checkout directory name under SRC_DIR
#                                    (default: hipcollections).
#   HIPCOLLECTIONS_BUILD_TYPE        build | install (default: build).
#   HIPCOLLECTIONS_CONDA_ENV         Conda environment to activate first.
#   HIPCOLLECTIONS_USE_VERSIONS_JSON 'true' resolves CPM dependencies through
#                                    the project's versions.json instead of the
#                                    sibling checkouts under SRC_DIR.
#   HIPCOLLECTIONS_INSTALL_COMPONENTS
#                                    Components installed when the build type is
#                                    'install'. Empty (the default) installs
#                                    everything; the project only defines the
#                                    'devel' component.
#   HIPCOLLECTIONS_CPACK_GENERATORS  Semicolon separated CPack generator list.
#                                    Defaults to the generators whose backend is
#                                    present (RPM and/or DEB).
#   HIPCOLLECTIONS_BUILD_TESTS       ON (default) builds the test targets.
#   HIPCOLLECTIONS_BUILD_BENCHMARKS  ON (default) builds the benchmarks.
#   HIPCOLLECTIONS_BUILD_EXAMPLES    ON (default) builds the examples.
#   CI_INTERNAL_CREATE_TESTS_TARBALL '1' packs the tests tarball the tester
#                                    images consume (default: 0).
#   CMAKE_INSTALL_PREFIX             Install prefix, overriding INSTALL_DIR; the
#                                    literal '<conda-prefix>' resolves to
#                                    CONDA_PREFIX.
#   CPACK_PACKAGING_INSTALL_PREFIX   Prefix baked into the packages. Defaults to
#                                    the install prefix.
#   RAPIDS_CMAKE_MODULE_PATH         Where the rocmds-cmake modules live, either
#                                    an installed package or a source checkout;
#                                    when set, they are used instead of a
#                                    download.
#   AMDGPU_TARGETS                   GPU architectures to compile for.
#   SCCACHE_ENABLE                   'true' enables the sccache launchers.
#   CMAKE_VERSION                    Version of the cmake package to install
#                                    into the venv under BUILD_DIR that supplies
#                                    cmake, cpack and ctest. Unset means
#                                    whatever cmake is on PATH, e.g. from a
#                                    conda environment.
#   ROCM_PATH, CMAKE_PREFIX_PATH
#   CMAKE_BUILD_PARALLEL_LEVEL, MAX_JOBS   Build parallelism.

set -xeu

: "${SRC_DIR:?SRC_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${INSTALL_DIR:?INSTALL_DIR must be set}"
: "${BUILD_ARTIFACTS_DIR:?BUILD_ARTIFACTS_DIR must be set}"

PROJECT_ID="${PROJECT_ID:-hipcollections}"

[[ "${HIPCOLLECTIONS_USE_VERSIONS_JSON:-false}" == "false" ]] && no_use_versions_json="1"
[[ "${HIPCOLLECTIONS_BUILD_TYPE:-build}" == "install" ]] && do_install="1"

if [ -n "${HIPCOLLECTIONS_CONDA_ENV:-}" ]; then
  set +u # note: conda script may have unbound variables
  source ${CONDA_DIR}/etc/profile.d/conda.sh
  conda activate ${HIPCOLLECTIONS_CONDA_ENV}
  set -u
fi

# The project installs everything under the 'devel' component; an empty list
# means a full install, which amounts to the same set of files.
components="${HIPCOLLECTIONS_INSTALL_COMPONENTS-}"
tests_tarball_name="${PROJECT_ID}-tests.tar.gz"

case "${CI_INTERNAL_CREATE_TESTS_TARBALL:-0}" in
  1 | true | yes | on) create_tests_tarball="1" ;;
  *) create_tests_tarball="" ;;
esac

[[ "${CMAKE_INSTALL_PREFIX:-}" == "<conda-prefix>" ]] && install_to_conda_prefix="1"
[[ "${SCCACHE_ENABLE:-false}" == "true" ]] && use_sccache=1

if [ -d "/opt/rh/gcc-toolset-$(g++ -dumpversion)" ]; then
  toolchain="/opt/rh/gcc-toolset-$(g++ -dumpversion)/root/usr"
  export CCC_OVERRIDE_OPTIONS="+--gcc-toolchain=${toolchain}"
  # note: in some shells, cmake is using /bin/{cc, c++} as compilers,
  # which may not be the compilers of the enabled toolchain.
  export CC=${toolchain}/bin/cc
  export CXX=${toolchain}/bin/c++
fi

#### packaging backends

if [ -n "${HIPCOLLECTIONS_CPACK_GENERATORS:-}" ]; then
  cpack_generators="${HIPCOLLECTIONS_CPACK_GENERATORS}"
else
  detected=()
  if command -v rpmbuild >/dev/null 2>&1; then
    detected+=("RPM")
  fi
  if command -v dpkg-deb >/dev/null 2>&1; then
    detected+=("DEB")
  fi
  cpack_generators="$(IFS=';'; echo "${detected[*]:-}")"
fi

if [ -z "${cpack_generators}" ]; then
  echo "ERROR: no CPack backend available (need rpmbuild and/or dpkg-deb)" >&2
  exit 2
fi

#### configure

# INSTALL_CUCO is what gates include(CPack), so it has to be on for the
# 'package' target to exist at all. INSTALL_CUCO_CPM_DEPS is off because the
# ROCm-DS stack packages the CPM dependencies (libhipcxx in particular)
# separately; that also selects the hipcollections-dev/-devel package names
# over the self-contained hipcollections-standalone one.
cmake_args=(
  "-D BUILD_TESTS=${HIPCOLLECTIONS_BUILD_TESTS:-ON}"
  "-D BUILD_BENCHMARKS=${HIPCOLLECTIONS_BUILD_BENCHMARKS:-ON}"
  "-D BUILD_EXAMPLES=${HIPCOLLECTIONS_BUILD_EXAMPLES:-ON}"
  "-D INSTALL_CUCO=ON"
  "-D INSTALL_CUCO_CPM_DEPS=OFF"
  "-D CMAKE_VERBOSE_MAKEFILE=ON"
  # Catch2 comes from the gcc toolset the wrapper exports, while the HIP test
  # executables are linked by clang, which defaults to -pie.
  "-D CMAKE_POSITION_INDEPENDENT_CODE=ON"
  "-D CPACK_OUTPUT_FILE_PREFIX=${BUILD_ARTIFACTS_DIR}"
  "-D CPACK_GENERATOR=${cpack_generators}"
  ${no_use_versions_json:+"-D CPM_libhipcxx_SOURCE=${SRC_DIR}/libhipcxx"}
  ${no_use_versions_json:+"-D CPM_hipbench_SOURCE=${SRC_DIR}/hipbench"}
  ${no_use_versions_json:+"-D CPM_nvbench_SOURCE=${SRC_DIR}/hipbench"}
  ${CMAKE_PREFIX_PATH:+"-D CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}${ROCM_PATH:+:${ROCM_PATH}/lib/cmake}"}
  ${AMDGPU_TARGETS:+"-D CMAKE_HIP_ARCHITECTURES=${AMDGPU_TARGETS}"}
  ${use_sccache:+"-D CMAKE_HIP_COMPILER_LAUNCHER=sccache"}
  ${use_sccache:+"-D CMAKE_CXX_COMPILER_LAUNCHER=sccache"}
  ${use_sccache:+"-D CMAKE_C_COMPILER_LAUNCHER=sccache"}
)

if [ -n "${install_to_conda_prefix:-}" ]; then
  cmake_args+=(
    "-D CMAKE_INSTALL_PREFIX=${CONDA_PREFIX}"
    "-D CPACK_PACKAGING_INSTALL_PREFIX=${CONDA_PREFIX}"
  )
else
  install_prefix="${CMAKE_INSTALL_PREFIX:-${INSTALL_DIR}}"
  package_prefix="${CPACK_PACKAGING_INSTALL_PREFIX:-${install_prefix}}"
  cmake_args+=(
    "-D CMAKE_INSTALL_PREFIX=${install_prefix}"
    "-D CPACK_PACKAGING_INSTALL_PREFIX=${package_prefix}"
  )
fi

build_args=(
  "--parallel ${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS:-1}}"
)

src_dir=${SRC_DIR}/${PROJECT_ID}
build_src_dir=${BUILD_DIR}/${PROJECT_ID}
build_build_dir=${BUILD_DIR}/${PROJECT_ID}/build

mkdir -p ${build_src_dir}
rm -rf ${build_src_dir} # NOTE: not necessary but helpful for debugging
cp -R ${src_dir} ${build_src_dir}

mkdir -p ${BUILD_ARTIFACTS_DIR}

# cmake, cpack and ctest come from this venv.
if [ -n "${CMAKE_VERSION:-}" ]; then
  python3 --version
  python3 -m venv ${build_src_dir}/_venv
  . ${build_src_dir}/_venv/bin/activate
  pip3 install --upgrade pip
  pip3 install "cmake==${CMAKE_VERSION}"
fi
cmake --version && cpack --version && ctest --version

#### build, package, install

cmake -S ${build_src_dir} -B ${build_build_dir} ${cmake_args[@]}
cmake --build ${build_build_dir} ${build_args[@]}
cmake --build ${build_build_dir} --target package ${build_args[@]}

if [ -n "${do_install:-}" ]; then
  if [ -n "${components}" ]; then
    for component in ${components}; do
      cmake --install ${build_build_dir} --component ${component} --verbose
    done
  else
    cmake --install ${build_build_dir} --verbose
  fi
fi

#### test archive

# The tester images extract this at / and drive ci/internal/test.sh out of the
# source tree it carries, so the archive keeps absolute paths. Unlike hipbench
# and jitify, the configured source copy under BUILD_DIR stays out: the tester
# runs prebuilt binaries through ctest rather than recompiling, and the only
# test data it reads (roaring bitmaps) is downloaded into the build tree.
if [ -n "${create_tests_tarball:-}" ]; then
  tar --exclude=.git -czvf ${BUILD_ARTIFACTS_DIR}/${tests_tarball_name} ${build_build_dir} ${src_dir}
  du -sh ${BUILD_ARTIFACTS_DIR}/${tests_tarball_name}
fi

# inspect package contents
if command -v dpkg >/dev/null 2>&1; then
  find ${BUILD_ARTIFACTS_DIR} -name "*.deb" -exec echo "{}:" \; -exec dpkg -c {} \;
fi
if command -v rpm >/dev/null 2>&1; then
  find ${BUILD_ARTIFACTS_DIR} -name "*.rpm" -exec echo "{}:" \; -exec rpm -qlp {} \;
fi
