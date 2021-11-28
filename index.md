## [<i>MRS 2021 Fall Meeting Tutorial:</i>](https://www.mrs.org/meetings-events/fall-meetings-exhibits/2021-mrs-fall-meeting/call-for-papers/tutorial-sessions) <br> [Symmetry-Aware Neural Networks<br>for the Material Sciences with](https://www.mrs.org/meetings-events/fall-meetings-exhibits/2021-mrs-fall-meeting/call-for-papers/tutorial-sessions-detail/2021_mrs_fall_meeting/eq04/tutorial-eq04-) [`e3nn`](https://e3nn.org)

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


## Spacetime coordinates {#spacetime}

<pre>
Monday, November 29, 2021
8:30 AM - 5:00 PM
Hynes, Level 2, Room 205
</pre>
[Hynes Convention Center Floor Plans](https://www.signatureboston.com/hynes/floor-plans-and-specs)

### Schedule

8:30 - 9:30am  | <a href="#tut1">Tutorial 1: Intro to Symmetry in ML</a>
9:30 - 10:00am | BREAK
10:00 - 11:00am | <a href="#tut2">Tutorial 2: Group theory, irreps, and tensor products</a> 
11:00 - noon | <a href="#tut3">Tutorial 3: Invariant functions for geometry of atomic structures</a>
noon - 1:30pm | LUNCH
1:30-2:30pm | <a href="#tut4">Tutorial 4: Molecular dynamics</a>
2:30-3:00pm | BREAK
3:00-4:00pm | <a href="#tut5">Tutorial 5: Electron densities</a>
4:00-5:00pm | <a href="#tut6">Tutorial 6: Phonon properties</a>


## Tutorials and Materials {#tutorials}

### Overview 

### Tutorial 1: Euclidean Symmetry in Machine Learning for Materials Science by <i>Tess Smidt</i> {#tut1}
Video | Slides | Colab | Code

### Tutorial 2: Group theory, irreducible representations, and tensor products and how to use them in `e3nn` to build Euclidean Neural Networks by <i>Mario Geiger</i> {#tut2}
Video | Slides | Colab | Code

### Tutorial 3: Analyzing geometry and structure of atomic configurations with equivariant and invariant functions by <i>Martin Uhrin</i> {#tut3}
Video | Slides | Colab | Code

### Tutorial 4: Molecular dynamics with NequIP by <i>Simon Batzner and Alby Musaelian</i> {#tut4}
Video | Slides | [Colab](https://colab.research.google.com/drive/1_r348f6oIyKxH4FnpKeD8g4QjwDhP8mT?usp=sharing) | [NequIP](https://github.com/mir-group/nequip)

### Tutorial 5: Predicting Electron Densities with e3nn by <i>Josh Rackers</i> {#tut5}
Video | Slides | [Colab](https://colab.research.google.com/drive/1ryOQ6hXxCidM_mGN0Yrf4BbjUtpyCxgy?usp=sharing) | Code

### Tutorial 6: Predicting Phonon Properties of Crystal Structures by <i>Nina Andrejevic and Zhantao Chen</i> {#tut6}
Video | Slides | Colab | Code

## Instructors

### [Tess Smidt](https://blondegeek.net) (tsmidt@mit.edu)
Tess Smidt is an Assistant Professor of Electrical Engineering and Computer Science at MIT. Tess earned her SB in Physics from MIT in 2012 and her PhD in Physics from the University of California, Berkeley in 2018. Her research focuses on machine learning that incorporates physical and geometric constraints, with applications to materials design. Prior to joining the MIT EECS faculty, she was the 2018 Alvarez Postdoctoral Fellow in Computing Sciences at Lawrence Berkeley National Laboratory and a Software Engineering Intern on the Google Accelerated Sciences team where she developed Euclidean symmetry equivariant neural networks which naturally handle 3D geometry and geometric tensor data.

### [Mario Geiger](https://mariogeiger.ch/) (geiger.mario@gmail.com)
Mario Geiger is a PhD in the laboratory of Physics of Complex Systems at EPFL in Switzerland. He is a physicist and he studies the dynamics of neural networks. He also studies the theory of equivariant neural networks and is the main developer of e3nn, a library for neural networks aware of the Euclidean symmetries.

### Martin Uhrin
Martin Uhrin is a senior postdoctoral research in the group of Nicola Marzari at EPFL in Switzerland. The focus of his research is the development and application of methods that accelerate materials discovery, particularly in the area of prediction and rationalisation of crystal structures. To this end, Martin has built up expertise in ab-initio structure prediction, empirical potential development, cluster prediction and deep learning. He has also contributed to the development of many community tools and was lead architect of the AiiDA workflow engine, part of the widely used AiiDA materials informatics platform.
Martin earned his Ph.D. in computational physics from the University College London under the supervision of Chris Pickard. He holds an MPhys. in computational physics from the University of Edinburgh.

### Simon Batzner
Simon Batzner is a PhD student at Harvard University, working under the supervision of Prof. Boris Kozinsky. His research interests focus on the development of Machine-Learning Interatomic Potentials for accelerated Molecular Dynamics simulations. Before joining Harvard, he obtained his Master’s at MIT, where he worked with Prof. Alexie Kolpak on problems in atomistic Machine Learning. He obtained his bachelor’s degree at the University of Stuttgart, Germany, during which he also spent one year working on the SOFIA mission at the NASA Armstrong Flight Research Center. 

### Alby Musaelian

### Josh Rackers
Josh Rackers is a Truman Fellow at Sandia National Labs. Josh is interested in developing methods to enable quantum-accurate simulation of biological molecules. To this end, he is involved in research spanning from quantum chemistry through classical molecular dynamics simulation. Josh is particularly interested in using physics-aware machine learning (ML) models to predict charge densities of large biomolecular systems. Josh earned a Ph.D. in Biophysics from Washington University in St. Louis and a B.S. in Physics and Political Science from Ohio State University. Before venturing into graduate school, he taught high school physics and chemistry in Baltimore, Maryland. 


### Nina Andrejevic
Nina Andrejevic is a PhD student at the Massachusetts Institute of Technology working with Prof. Mingda Li in the Quantum Matter Group. Nina’s research interests include investigating electronic and phononic properties of emergent quantum materials using neutron and X-ray spectroscopies, as well as developing complementary quantum theories and machine learning methods to help interpret and extract key insights from experiments.

### Zhantao Chen
Zhantao Chen is a PhD student at the Massachusetts Institute of Technology working with Prof. Mingda Li in the Quantum Matter Group and Prof. Jing Kong. Zhantao’s research interests include merging machine learning methods into computational physics to help extract knowledge from scattering measurements.
