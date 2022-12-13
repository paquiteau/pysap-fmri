========
fMRI-sim
========

A simulator of synthetic fMRI data, in order to have a controllable input data to validate the performance of reconstruction


Using the simulator
===================

The simulator comes with several Scenario already available

.. code-block:: python

    from fmri_sim import  get_scenario_data()

    scenario_data = get_scenario_data("shepp-block")

However you can also create a Custom scenario:


.. code-block:: python

    >>> import matplotlib.pyplot as plt
    >>> from fmri_sim import CustomPhantomScenario

    >>> scn = CustomPhantomScenario(shape=(128,128), n_frames=100, type="shepp-logan")
    >>> phantom = scn.get_phantom()
    >>> plt.imshow(phantom[0, ... ]) # the time is the first axis.
    >>> scn.add_blocks(roi="auto", decay=None, )
    >>> scn.add_acquisition(protocol="EPI", n_coils=16)
    >>> scn.get_build_steps() # print all the build steps
    Build steps:
    [(create_phantom, {shape:(128, 128), "n_frames":100, "type":shepp-logan"}),
    (add_block, {}]

    >>>scn.add_function

    scn.get_data() # return all the data available for the reconstruction


Underlying: The phantom dataclass, should be immutable ?

Other fMRI data simulator
=========================

neuroRsim_
    In R, target statistical method validation.

fMRIsim_
    In Python, focuses on the noise statistics and validation of statistical analysis, heavily inspired by neuroRsim

POSSUM_
   In Python, part of the FSL Toolbox. It models a wide range of data inhomogeinity, going back to the Block equations.
   It can be use to augment existing data.


.. _neuroRsim: https://www.jstatsoft.org/article/view/v044i10
.. _fMRIsim: https://brainiak.org/docs/brainiak.utils.html#module-brainiak.utils.fmrisim
.. _POSSUM: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/POSSUM
