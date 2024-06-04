#!/bin/bash
# Copyright (c) 2018-2022, NVIDIA CORPORATION.
##############################

# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# hipCollections Style Tester #
##############################

# Ignore errors and set path
set +e
#todo: activate if system-wide conda installation is provided
if ! [ -z ${CONDA_PREFIX_1} ]; then
PATH=${CONDA_PREFIX_1}:$PATH
else
PATH=/conda/bin:$PATH
fi

# LC_ALL=C.UTF-8
# LANG=C.UTF-8

# Activate common conda env
if ! [ -z ${CONDA_PREFIX_1} ]; then
. ${CONDA_PREFIX_1}/etc/profile.d/conda.sh
else
. /opt/conda/etc/profile.d/conda.sh
fi
conda activate rapids

# Run clang-format and check for a consistent code format
CLANG_FORMAT=`pre-commit run clang-format --all-files 2>&1`
CLANG_FORMAT_RETVAL=$?

# Run doxygen check
DOXYGEN_CHECK=`ci/checks/doxygen.sh`
DOXYGEN_CHECK_RETVAL=$?

echo -e "$DOXYGEN_CHECK"

RETVALS=(
  $CLANG_FORMAT_RETVAL
)
IFS=$'\n'
RETVAL=`echo "${RETVALS[*]}" | sort -nr | head -n1`

exit $RETVAL
