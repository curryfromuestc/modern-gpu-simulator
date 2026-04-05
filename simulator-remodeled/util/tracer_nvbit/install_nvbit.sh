export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

rm -rf $BASH_ROOT/nvbit_release
wget https://github.com/NVlabs/NVBit/releases/download/v1.7.5/nvbit-Linux-aarch64-1.7.5.tar.bz2
tar -xf nvbit-Linux-aarch64-1.7.5.tar.bz2 -C $BASH_ROOT
rm nvbit-Linux-aarch64-1.7.5.tar.bz2
mv $BASH_ROOT/nvbit_release_aarch64 $BASH_ROOT/nvbit_release


