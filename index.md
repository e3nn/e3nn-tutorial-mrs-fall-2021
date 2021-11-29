# [<i>MRS 2021 Fall Meeting Tutorial:</i>](https://www.mrs.org/meetings-events/fall-meetings-exhibits/2021-mrs-fall-meeting/call-for-papers/tutorial-sessions) <br> [Symmetry-Aware Neural Networks<br>for the Materials Science with](https://www.mrs.org/meetings-events/fall-meetings-exhibits/2021-mrs-fall-meeting/call-for-papers/tutorial-sessions-detail/2021_mrs_fall_meeting/eq04/tutorial-eq04-) [`e3nn`](https://e3nn.org)

Understanding the role of symmetry in the physical sciences is critical for choosing an appropriate machine-learning method.

In this tutorial we will present how to incorporate Euclidean and permutation symmetries into neural networks and the benefits of these techniques for predicting a variety of properties of atomic systems (molecules, crystal, and beyond).
* The morning session will include..
   * [an overview of symmetry in machine learning and materials science](#tut1)
   * [an introduction to group representation theory and the implementation of symmetry equivariant operations in the open-source PyTorch framework for Euclidean neural networks](#tut2) [`e3nn`](https://e3nn.org)
   * [demonstration of how equivariant operations can create rich descriptions of atomic geometry](#tut3)
* The afternoon session will demonstrate specific use cases of symmetry-aware methods for diverse applications
   * [molecular dynamics](#tut4) with [NequIP](https://github.com/mir-group/nequip)
   * [electron densities](#tut5)
   * [phonon properties](#tut6).

We will provide a cloud-based notebook environment for participants to run the code example used throughout the tutorial. Participants will leave the tutorial with theory and code resources in hand and a practical working knowledge of how symmetry considerations impact algorithm design in machine learning and beyond.


# Spacetime coordinates {#spacetime}

<pre>
Monday, November 29, 2021
8:30 AM - 5:00 PM
Hynes, Level 2, Room 205
</pre>
[Hynes Convention Center Floor Plans](https://www.signatureboston.com/hynes/floor-plans-and-specs)

## Schedule

8:30 - 9:30am  | <a href="#tut1">Tutorial 1: Intro to Symmetry in ML</a>
9:30 - 10:00am | BREAK
10:00 - 11:00am | <a href="#tut2">Tutorial 2: Group theory, irreps, and tensor products</a>
11:00 - noon | <a href="#tut3">Tutorial 3: Invariant functions for geometry of atomic structures</a>
noon - 1:30pm | LUNCH
1:30-2:30pm | <a href="#tut4">Tutorial 4: Molecular dynamics</a>
2:30-3:00pm | BREAK
3:00-4:00pm | <a href="#tut5">Tutorial 5: Electron densities</a>
4:00-5:00pm | <a href="#tut6">Tutorial 6: Phonon properties</a>


# Tutorials and Materials {#tutorials}

[YouTube playlist for tutorial](https://www.youtube.com/watch?v=q9EwZsHY1sk&list=PLx3xbphkO3qIlBoESkbafXaDtr0tq5iRd)

## Tutorial 1: Euclidean Symmetry in Machine Learning for Materials Science by [<i>Tess Smidt</i>](#tess) {#tut1}
[Video](https://youtu.be/q9EwZsHY1sk) | [Slides](https://docs.google.com/presentation/d/1y-fCZihLoSBgAqNYKZY1-jy8qpvqagqPTsX-jR0uTm8/edit?usp=sharing)
 
## Tutorial 2: Group theory, irreducible representations, and tensor products and how to use them in `e3nn` to build Euclidean Neural Networks by [<i>Mario Geiger</i>](#mario) {#tut2}
[Video](https://youtu.be/9rS8gtey_Ic) | [Slides](https://slides.com/mariogeiger/e3nn_mrs_2021/) 

* [Colab for Spherical Harmonics](https://colab.research.google.com/drive/1JYlgVk68dHb8IEHsOHtT1dDNdOohJD00?usp=sharing)
* [Colab for Tensor Products with weights](https://colab.research.google.com/drive/1aR2HuZvKbvVNVUUjDVRl1ne9D2Kag7Mn?usp=sharing)
* [Colab for Reduce Tensor Product](https://colab.research.google.com/drive/1SYRLJS2MPmRyguXn3RZ_L5CoR3sLgW8w?usp=sharing)

## Tutorial 3: Analyzing geometry and structure of atomic configurations with equivariant and invariant functions by [<i>Martin Uhrin</i>](#martin) and [<i>Thomas Hardin</i>](#thomas) {#tut3}
Video | Slides (see `notebooks` in Code) | [Colab](https://colab.research.google.com/drive/1duR1Y-roE_CSL3hrGxINMZ4XIepHvoHt?usp=sharing) | [Code](https://github.com/muhrin/mrs-tutorial)

## Tutorial 4: Molecular dynamics with NequIP by [<i>Simon Batzner</i>](#simon) and [<i>Alby Musaelian</i>](#alby) {#tut4}
Video | Slides | [Colab](https://colab.research.google.com/drive/1_r348f6oIyKxH4FnpKeD8g4QjwDhP8mT?usp=sharing) | [NequIP](https://github.com/mir-group/nequip)

## Tutorial 5: Predicting Electron Densities with e3nn by [<i>Josh Rackers</i>](#josh) {#tut5}
Video | Slides | [Colab](https://colab.research.google.com/drive/1ryOQ6hXxCidM_mGN0Yrf4BbjUtpyCxgy?usp=sharing) | Code

## Tutorial 6: Predicting Phonon Properties of Crystal Structures by [<i>Nina Andrejevic</i>](#nina) and [<i>Zhantao Chen</i>](#zhantao) {#tut6}
Video | Slides | Colab | [Code](https://github.com/ninarina12/phononDoS_tutorial)

# Related Talks at MRS 2021 Fall Meeting {#related}

## [EQ04.01.03](https://www.mrs.org/meetings-events/fall-meetings-exhibits/2021-mrs-fall-meeting/call-for-papers/symposium-sessions-detail/2021_mrs_fall_meeting/eq04)<br>Using Complete, Symmetry Invariant Representations of Atomic Environments to Predict New Ionic Liquids from Experimental Data <br><br> [<i>Martin Uhrin</i>](#martin)

<pre>
Tuesday 11 AM, November 30, 2021
Hynes, Level 2, Room 205
</pre>

The ultimate goal of any atomistic design process, be it for materials or molecules, is to start with one or more target properties and predict atomic geometries that are statistically likely to possess these properties. In this talk, I will discuss how to construct complete, symmetry invariant descriptions of atomic environments that can be inverted to recover the original environment. Using this representation I will show that it is possible to build a model that can predict the conductivity of ionic liquids, an important potential electrolyte for future battery technologies. By training on a dataset of experimental results the model learns how to attribute contributions to the overall conductivity to fragments of the anion and cation. This allows us to virtually screen thousands of anion/cation combinations to pinpoint those with high conductivities. Furthermore, it opens the door to designing entirely new molecules by combining fragments to achieve properties that go beyond those that are possible using current ionic liquids.

## [DS03.06.03](https://www.mrs.org/meetings-events/fall-meetings-exhibits/2021-mrs-fall-meeting/call-for-papers/symposium-sessions-detail/2021_mrs_fall_meeting/ds03)<br>NequIP—E(3)-Equivariant Convolutions Enable Sample-Efficient, Scalable and Highly Accurate Machine Learning Interatomic Potentials<br><br><i> [Simon Batzner](#simon), [Albert Musaelian](#alby), Lixin Sun, [Tess Smidt](#tess), [Mario Geiger](#mario), Jonathan Mailoa, Mordechai Kornbluth, Nicola Molinari, Boris Kozinsky</i>

<pre>
Thursday 1:45 PM, November 30, 2021
Sheraton, 5th Floor, The Fens
</pre>

We present Neural Equivariant Interatomic Potentials (NequIP), an E(3)-equivariant deep learning approach for learning interatomic potentials for molecular dynamics simulations. Instead of the commonly deployed invariant convolutions over scalar features, NequIP uses E(3)-equivariant convolutions over geometric tensors, better representing the symmetries of Euclidean space. The proposed model obtains state-of-the-art accuracy on a challenging set of diverse molecules and materials while at the same exhibiting remarkable sample efficiency. Interestingly, NequIP outperforms existing, invariant models with up to three orders of magnitude fewer training data and performs better than kernel methods, even on tiny data sets, thereby challenging the widely held belief that deep neural networks require massive training sets. We show results from molecular dynamics simulations using NequIP on a series of technologically relevant bulk materials as well as the folding of small proteins. The method is implemented in a scalable and highly efficient software implementation, integrated with the molecular dynamics code LAMMPS, and can be used to simulate large time- and length-scales at high accuracy and low computational cost.

## [DS03.06.02](https://www.mrs.org/meetings-events/fall-meetings-exhibits/2021-mrs-fall-meeting/call-for-papers/symposium-sessions-detail/2021_mrs_fall_meeting/ds03)<br>DICE—A Linear-Scaling N-Body Interatomic Potential from E(3)-Equivariant Convolutions<br><br><i>[Albert Musaelian](#alby), [Simon Batzner](#simon), Lixin Sun, Steven Torrisi, Boris Kozinsky</i>
<pre>
Thursday 2:00 PM, November 30, 2021
Sheraton, 5th Floor, The Fens
</pre>

Message Passing Graph Neural Networks (MPNNs) based on pairwise interactions have emerged as the leading paradigm for modeling atomistic systems by recursively propagating information along a molecular graph. While MPNNs have consistently been demonstrated to give low generalization errors, they inherently have a low level of interpretability, are not systematically improvable, and are difficult to scale to large numbers of atoms. Here, we introduce the Deep Interatomic Cluster Expansion (DICE), an equivariant neural network that leverages many-body information in a single interaction, without the need for message passing or convolutions. The method can be systematically improved by including higher-order interactions at linear cost, has physically meaningful hyperparameters, and is embarrassingly parallel. DICE builds on a novel, learnable E(3)-equivariant many-body representation that utilizes weighted tensor products of geometric features to describe N-point correlations of atoms. The proposed many-body representation overcomes the exponential scaling of a naive cluster expansion and instead scales linearly with the number of simultaneously correlated particles. We demonstrate that the use of higher-order correlations of atoms systematically improves the accuracy. We further find that DICE gives excellent performance across a wide variety of settings, outperforming both MPNNs as well as kernel-based methods on small data sets.

# Instructors

## [Tess Smidt](https://blondegeek.net) (tsmidt@mit.edu) {#tess}
Tess Smidt is an Assistant Professor of Electrical Engineering and Computer Science at MIT. Tess earned her SB in Physics from MIT in 2012 and her PhD in Physics from the University of California, Berkeley in 2018. Her research focuses on machine learning that incorporates physical and geometric constraints, with applications to materials design. Prior to joining the MIT EECS faculty, she was the 2018 Alvarez Postdoctoral Fellow in Computing Sciences at Lawrence Berkeley National Laboratory and a Software Engineering Intern on the Google Accelerated Sciences team where she developed Euclidean symmetry equivariant neural networks which naturally handle 3D geometry and geometric tensor data.

## [Mario Geiger](https://mariogeiger.ch/) (geiger.mario@gmail.com) {#mario}
Mario Geiger is a PhD in the laboratory of Physics of Complex Systems at EPFL in Switzerland. He is a physicist and he studies the dynamics of neural networks. He also studies the theory of equivariant neural networks and is the main developer of e3nn, a library for neural networks aware of the Euclidean symmetries.

## Martin Uhrin {#martin}
Martin Uhrin is a senior postdoctoral research in the group of Nicola Marzari at EPFL in Switzerland. The focus of his research is the development and application of methods that accelerate materials discovery, particularly in the area of prediction and rationalisation of crystal structures. To this end, Martin has built up expertise in ab-initio structure prediction, empirical potential development, cluster prediction and deep learning. He has also contributed to the development of many community tools and was lead architect of the AiiDA workflow engine, part of the widely used AiiDA materials informatics platform.
Martin earned his Ph.D. in computational physics from the University College London under the supervision of Chris Pickard. He holds an MPhys. in computational physics from the University of Edinburgh.

## Thomas Hardin {#thomas}
Thomas Hardin is a Truman Fellow at Sandia National Laboratories. Thomas designs disordered materials and uses machine learning to discover low-dimensional manifolds underpinning the structure of glass. His other research work with disordered materials includes mesoscale and atomistic modeling, additive manufacturing, and microparticle impact testing. Thomas earned his doctorate degree in Materials Science and Engineering from the Massachusetts Institute of Technology, working with Professor Chris Schuh, and a dual bachelor’s degree in Mathematics and Mechanical Engineering from Brigham Young University, working with Professors Brent Adams and Eric Homer.

## Simon Batzner {#simon}
Simon Batzner is a PhD student at Harvard University, working under the supervision of Prof. Boris Kozinsky. His research interests focus on the development of Machine-Learning Interatomic Potentials for accelerated Molecular Dynamics simulations. Before joining Harvard, he obtained his Master’s at MIT, where he worked with Prof. Alexie Kolpak on problems in atomistic Machine Learning. He obtained his bachelor’s degree at the University of Stuttgart, Germany, during which he also spent one year working on the SOFIA mission at the NASA Armstrong Flight Research Center.

## Alby Musaelian {#alby}
Albert Musaelian is a PhD student in Boris Kozinsky's Materials Intelligence Research (MIR) group at Harvard University. He works on the design and implementation of equivariant machine learning methods for atomistic simulations with a particular focus on machine-learning interatomic potentials.

## Josh Rackers {#josh}
Josh Rackers is a Truman Fellow at Sandia National Labs. Josh is interested in developing methods to enable quantum-accurate simulation of biological molecules. To this end, he is involved in research spanning from quantum chemistry through classical molecular dynamics simulation. Josh is particularly interested in using physics-aware machine learning (ML) models to predict charge densities of large biomolecular systems. Josh earned a Ph.D. in Biophysics from Washington University in St. Louis and a B.S. in Physics and Political Science from Ohio State University. Before venturing into graduate school, he taught high school physics and chemistry in Baltimore, Maryland.


## Nina Andrejevic {#nina}
Nina Andrejevic is a PhD student at the Massachusetts Institute of Technology working with Prof. Mingda Li in the Quantum Matter Group. Nina’s research interests include investigating electronic and phononic properties of emergent quantum materials using neutron and X-ray spectroscopies, as well as developing complementary quantum theories and machine learning methods to help interpret and extract key insights from experiments.

## Zhantao Chen {#zhantao}
Zhantao Chen is a PhD student at the Massachusetts Institute of Technology working with Prof. Mingda Li in the Quantum Matter Group and Prof. Jing Kong. Zhantao’s research interests include merging machine learning methods into computational physics to help extract knowledge from scattering measurements.
