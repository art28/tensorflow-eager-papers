import wget
import os
import tarfile

file_path = os.path.dirname(os.path.abspath(__file__))

print("Download cifar Dataset")
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
filename = wget.download(url)

print("extract tar file")

tar = tarfile.open(filename)
tar.extractall()
tar.close()

new_path = os.path.join(file_path, "data/cifar-10-batches-py/")

print("make new directory in %s" % new_path)

if not os.path.exists("data"):
    os.makedirs("data")


os.rename("cifar-10-batches-py", new_path)
os.remove(filename)

