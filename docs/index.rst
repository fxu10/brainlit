..  -*- coding: utf-8 -*-

.. _contents:

Overview of Brainlit
====================

Brainlit is a Python package for reading and analyzing brain data.
The brain data is assumed to consist of image files in an octree structure to handle mutliple resolutions.
Optionally, the package is able to handle skeletonized axon data stored in a `.swc` file format.

Brainlit is able to handle this data, visualizing and running analysis with morphological and statistical methods.
A diagram demonstrating the capabilities of the package is shown.

.. image:: images/figure.png
   :width: 600

Push/Pull Data
--------------

Brainlit uses the Cloudvolume package to handle data transfer, and modifies code to simplify data management.
Methods exist to pull, push and store data in multiple formats such as csv, octree, or swc.
The only needed parameters to specify are a filepath or url for input data and a filepath or url for output destination.
The package handles google cloud and aws destinations.

The tutorial can be found here.

Visualize
---------

Brainlit has multiple visualization tools including 2d and 3d visualization.
Interactive viewing of 3d volumes locally is built off of the Napari package.
An in-browser interactive viewer is possible via Neuroglancer.

The tutorial can be found here.

Manually Segment
----------------

Brainlit supports manual labeling of uploaded images via a custom pipeline class.
Usage is simple and multiple users can segment and upload different parts of the same image.

The tutorial can be found here.

Automatically/Semi-Automatically Segment
---------------------

Brainit allows for any automatic segmentation algorithms to be run on image data.
The package comes with built-in methods that can be used which leverage data provided by
skeleton ground truth data.

The tutorial can be found here.

Documentation
=============

.. toctree::
    :maxdepth: 1

    tutorial
    reference/index
    license

.. toctree::
    :maxdepth: 1
    :caption: Useful Links

    Brainlit @ GitHub <http://www.github.com/neurodata/brainlit/>
    Issue Tracker <https://github.com/neurodata/brainlit/issues>


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
