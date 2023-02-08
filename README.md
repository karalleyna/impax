## The JAX implementation of Local Deep Implicit Functions for 3D Shape
### ldif -> impax

Hello everyone, we are a project group from Machine Learning for 3D Geometry course, affiliated with Technical University of Munich. We did this as a course project and also to enable further developments in this area.

- Burak Ömür
- Aleyna Kara


Our aim was to port/reimplement [ldif](https://github.com/google/ldif) repository for JAX. We constantly used documentation/code from the provided repository. Our aim is to spread this 3D Shape Representation and Reconstruction task to multiple frameworks.

Below one can find our abstract to obtain more details about this repository.


### Abstract

Local Dense Implicit Functions are a new 3D shape representation that can accurately reconstruct the surface shape of an object. Similar to implicit surfaces, these functions map whether a given point is within the object it represents or not. LDIF aims to be generalizable for different types of objects and to be storage-wise and computationally efficient. As an input, it takes a 3D mesh or a depth image decomposes given 3D space into parts and determines implicit functions for them by learning. In the paper, they show the performance of their algorithm on a variety of object types, from human body shape models to objects like furniture and vehicles. 

The authors have provided the implementation of their model alongside the paper. It has been developed in Python with Tensorflow and has very limited PyTorch support. For our project, we have re-implemented the paper using JAX and tested it with ShapeNet.


## Please refer to [original repository](https://github.com/google/ldif) for implementation / paper and other related stuff.

