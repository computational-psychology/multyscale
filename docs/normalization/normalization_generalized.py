{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "dcbfab0e",
   "metadata": {},
   "source": [
    "# Rewrite Reuse Refactor\n",
    "## Reproducing Unit Tests using refactored ODOG formulation\n",
    "This Notebook introduces a new way of formulating the normalization step of the ODOG family of Models, and shows that this formulation is numerically equivalent to the original paper implementation, to establish continuity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b4e3519d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Third party libraries\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from PIL import Image\n",
    "import scipy\n",
    "\n",
    "# Import local module\n",
    "import multyscale"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b0ec8f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# %% Load example stimulus\n",
    "stimulus = np.asarray(Image.open(\"example_stimulus.png\").convert(\"L\"))\n",
    "\n",
    "# %% Parameters of image\n",
    "# visual extent, same convention as pyplot:\n",
    "visextent = (-16, 16, -16, 16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a1e9446d",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = multyscale.models.ODOG_RHS2007(shape=stimulus.shape, visextent=visextent)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e7655c54",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Frontend filterbank of ODOG implementation by Robinson et al. (2007)\n",
    "O, S = model.bank.filters.shape[:2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c7e7f6b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Filter the example stimulus\n",
    "filters_output = model.bank.apply(stimulus)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "436e1ca2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Weight individual filter(outputs) according to spatial size (frequency)\n",
    "filters_output = model.weight_outputs(filters_output)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "73904479",
   "metadata": {},
   "source": [
    "This preamble defines a filterbank output which is not already normalized by any function. The Traditional ODOG normalization function is multyscale/models.py ODOG, shown below"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c119c91",
   "metadata": {},
   "outputs": [],
   "source": [
    "normalized_outputs = model.normalize_outputs(filters_output)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a46964ec",
   "metadata": {},
   "source": [
    "We're comparing this to a new implementation which replaces the math of the original function by a more general form.  \n",
    "The traditional formulation for the normalization schema of ODOG can be formulated as:\n",
    "$$ F' = \\frac{f_{o',s',x,y}}{\\sqrt{\\frac{1}{XY}\\sum_{y=1}^{Y} \\sum_{x=1}^{X}(\\sum_{o=1}^{O}\\sum_{s=1}^{S} {w_{o',s',o,s} f_{o,s,x,y})^2}}}$$\n",
    "where $w_{o', s', o, s} =   \\begin{cases} \n",
    "      1 & o = o'  \\\\\n",
    "      0 & else\n",
    "   \\end{cases} \n",
    "$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ce890e25",
   "metadata": {},
   "outputs": [],
   "source": [
    "normalizers = model.normalizers(filters_output)\n",
    "\n",
    "RMS = model.normalizers_to_RMS(normalizers)\n",
    "\n",
    "normed2 = np.ndarray(filters_output.shape)\n",
    "for o,s in np.ndindex(filters_output.shape[:2]):\n",
    "    normed2[o,s] = filters_output[o,s] / RMS[o,s]\n",
    "\n",
    "assert(np.allclose(normalized_outputs,normed2))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "eb2b2064",
   "metadata": {},
   "source": [
    "## Implement RMS as a spatial filter"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "66718ec2",
   "metadata": {},
   "source": [
    "#### Can also divide by \"images\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0c9b348d",
   "metadata": {},
   "outputs": [],
   "source": [
    "i_RMS = np.tile(RMS.reshape(6,7,1,1),(1,1,1024,1024))\n",
    "\n",
    "norm_i_outputs = np.ndarray(filters_output.shape)\n",
    "for o, s in np.ndindex(filters_output.shape[:2]):\n",
    "    f = filters_output[o,s,...]\n",
    "    n = i_RMS[o,s]\n",
    "    norm_i_outputs[o, s] = f / n\n",
    "\n",
    "assert(np.allclose(normalized_outputs, norm_i_outputs))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "97f53a2f",
   "metadata": {},
   "source": [
    "This is equivalent to an equation that implements the averaging in the denominator as a linear filter, that scales the filter aggregate produced by the sum over scales/orientations. This changes the dimensionality of the denominator, but not actually the result of the division, since image by image division is executed pixel-wise\n",
    "\n",
    "$$ODOG: \\frac{f_{o^*,s^*}}{\\sqrt{G_{x,y}*(\\sum_{o=1}^{O}\\sum_{s=1}^{S} {w_{o',s',o,s} f_{o,s,x,y})^2}}}$$\n",
    "$w_{o,s}$ same as above,  \n",
    "$G_{x,y} = \\frac{1}{XY} = \\frac{1}{1024^2}$ everywhere."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "973d9d93",
   "metadata": {},
   "source": [
    "### Implement spatial mean as spatial filter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "274ab70c",
   "metadata": {},
   "outputs": [],
   "source": [
    "G = np.ones((1024*2,1024*2)) / 1024**2 # have doubly sized kernel so that pixel reaches each pixel in the average\n",
    "\n",
    "i_RMS2 = np.ones(filters_output.shape)\n",
    "o=3 # choose sample dimension\n",
    "s=4\n",
    "# calculate RMS using the filter\n",
    "norm = normalizers[o,s].copy()\n",
    "norm = norm ** 2\n",
    "mean = multyscale.filters.apply(norm, G, padval=0)\n",
    "i_RMS2[o,s] = i_RMS2[o,s]*np.sqrt(mean)\n",
    "# this replicates the just verified normalizing image\n",
    "assert(np.allclose(i_RMS2[o,s], i_RMS[o,s]))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "82795b62",
   "metadata": {},
   "source": [
    "By renaming the remaining sum in the denominator, we can find an abstract formulation of the RMS-Norm as a sequence of linear filters\n",
    "\n",
    " $$\\frac{f_{o^*,s^*}}{\\sqrt{G_{x,y}*(\\sum_{o=1}^{O}\\sum_{s=1}^{S} {w_{o',s',o,s} f_{o,s,x,y})^2}}}=\\frac{f_{o^*,s^*}}{\\sqrt{G * N^2}} = \\frac{f_{o^*,s^*}}{\\sqrt{G * (W \\cdot f)^2}}$$\n",
    " \n",
    " Which helps us develop terminology. The entire fraction is called the divisive norm on the filterbank-output $f$.  \n",
    " $G * N^2$ is the (local) energy, calculated as the convolution of $G$ the spatial weighting, and $N^2$ the Normcoefficient image squared.  \n",
    "The Normcoefficient Image can be calculated in turn by computing the Normpool $W$ dotproduct against the same $f$ that we are normalizing."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c597a480",
   "metadata": {},
   "source": [
    "Crucially the spatial weighting Filter G is $1/XY$ at each of its (2X,2Y) entries to assure that each pixel will take its average from the entire image. This requires that all edges be padded with the value 0, so that no unnecessary information gets included into the averaging\n",
    "   \n",
    "An implementation of this formulation can be found here:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "97891411",
   "metadata": {},
   "outputs": [],
   "source": [
    "def divisive_norm(filter_outputs, o, s):\n",
    "    G = spatial_weighting(filter_outputs.shape[-2:])\n",
    "    N = normcoeff(filter_outputs, o, s)**2\n",
    "    \n",
    "    z = multyscale.filters.apply(N, G, padval=0)\n",
    "    normed_filter = filter_outputs[o,s,:] / np.sqrt(z)\n",
    "    return normed_filter\n",
    "\n",
    "def normcoeff(filter_outputs, o_0, s_0): \n",
    "    w = normpool(filter_outputs.shape[0],filter_outputs.shape[1], o_0, s_0)\n",
    "    coeffs = np.tensordot(filter_outputs, w, axes=([0, 1], [0, 1]))\n",
    "    return coeffs\n",
    "\n",
    "def normpool(O, S, o_0, s_0): # w_{o,s}\n",
    "    ODOG = True\n",
    "    LODOG, FLODOG = False, False\n",
    "    w = np.ones(shape=(O,S))\n",
    "    for o in range(O):\n",
    "        for s in range(S):\n",
    "            if ODOG or LODOG:\n",
    "                if o==o_0:\n",
    "                    w[o,s] = 1/S\n",
    "                else:\n",
    "                    w[o,s] = 0\n",
    "\n",
    "            if FLODOG:\n",
    "                if o==o_0:\n",
    "                    w[o,s] = 1 #gaussian(s-s_0)\n",
    "                else:\n",
    "                    w[o,s] = 0\n",
    "    return w\n",
    "\n",
    "def spatial_weighting(filter_shape):\n",
    "    ODOG = True\n",
    "    LODOG, FLODOG = False, False\n",
    "    if ODOG:\n",
    "        G = np.ones([dim*2 for dim in filter_shape]) / np.prod(filter_shape)\n",
    "        \n",
    "    if LODOG:\n",
    "        G = 1 #gaussian(filter_shape, sigma)\n",
    "\n",
    "    if FLODOG:\n",
    "        G = 1 #gaussian(filter_shape, sigma=k*s)\n",
    "    return G"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "601f1394",
   "metadata": {},
   "source": [
    "# Regresssion test new formulation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bdb3590a",
   "metadata": {},
   "source": [
    "Our goal is now to establish numerical equivalence between the traditional formulation and the result of the newly reformatted and mathematically motivated code\n",
    "\n",
    "The original implementation is already tested against the matlab implementation of RHS_2007, so we know it is sound.\n",
    "  \n",
    "We will use this notebook to figure out: 1) if all of the units of our new implementation act as expected and 2) if the integration of those units is again comparable to RHS_2007"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "309a9c28",
   "metadata": {},
   "source": [
    "Note that in the module, all of the below functions will appear in multyscale/test, separated into fixtures and tests."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3a27717c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def filtershape():\n",
    "    return (1024,1024)\n",
    "\n",
    "def test_unit_spatial_weighting(filter_shape):\n",
    "    G = spatial_weighting(filter_shape)\n",
    "    assert np.allclose(G, 1/np.prod(filter_shape))\n",
    "    \n",
    "test_unit_spatial_weighting(filtershape())\n",
    "\n",
    "def filteroutput():\n",
    "    return np.random.random(size=(6,7,1024,1024))\n",
    "\n",
    "\n",
    "def test_unit_normcoeff(filter_outputs):\n",
    "    # use sum over scales formulation instead of linear filter\n",
    "    if input(\"Run full 6x7 analysis? [y/n] \")==\"y\":\n",
    "        N = np.zeros(shape=(6,7,1024,1024))\n",
    "        for o, s in np.ndindex(filter_outputs.shape[:2]):\n",
    "            for x,y in np.ndindex(filter_outputs.shape[2:]):\n",
    "                N[o, s, x, y] = np.sum(filter_outputs[o,:,x,y]) / filter_outputs.shape[1]\n",
    "            assert np.allclose(normcoeff(filter_outputs, o, s), N[o, s])\n",
    "\n",
    "            print(f\"dimension {o},{s} is sound\")\n",
    "    else:\n",
    "        print(\"only doing 5 dimensions chosen at random\")\n",
    "        N = np.zeros(shape=(6,7,1024,1024))\n",
    "        for i in range(5):\n",
    "            o, s = np.random.randint(0,6), np.random.randint(0,7)\n",
    "            for x,y in np.ndindex(filter_outputs.shape[2:]):\n",
    "                N[o, s, x, y] = np.sum(filter_outputs[o,:,x,y]) / filter_outputs.shape[1] # Does W.f properly produce sum_o,s[w_o,s . f_o,s]\n",
    "            assert np.allclose(normcoeff(filter_outputs, o, s), N[o, s])\n",
    "\n",
    "            print(f\"dimension {o},{s} is sound\")\n",
    "        \n",
    "\n",
    "print(\"unit test normcoeff\")\n",
    "test_unit_normcoeff(filteroutput())\n",
    "print(\"unit test divisive\")\n",
    "\n",
    "def test_unit_divisive(filter_outputs):\n",
    "    for i in range(5):\n",
    "        o, s = np.random.randint(0,6), np.random.randint(0,7)\n",
    "        G = spatial_weighting(filter_outputs.shape[-2:])\n",
    "        N = normcoeff(filter_outputs, o, s)**2\n",
    "        z = np.mean(N)  # does N*G properly calculate mean(N)?\n",
    "        normed_filter = filter_outputs[o,s,:] / np.sqrt(z)\n",
    "        assert np.allclose(divisive_norm(filter_outputs, o, s), normed_filter)\n",
    "\n",
    "        print(f\"dimension {o},{s} is sound\")\n",
    "test_unit_divisive(filteroutput())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "afc4623f",
   "metadata": {},
   "source": [
    "So as one can see, all of the steps agree with the mathematical formulation of ODOG. The next step will be, to analyze whether the new formulation agrees not only with the math, but also with the already verified original implementation that is present in the current codebase."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56300fe0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import scipy\n",
    "\n",
    "def normalize_multiple_outputs(filters_output):\n",
    "\n",
    "    normalized_outputs = np.ones(shape=filters_output.shape)\n",
    "    for o, s in np.ndindex(filters_output.shape[:2]):\n",
    "        normalized_outputs[o,s] = divisive_norm(filters_output, o,s)\n",
    "\n",
    "    return normalized_outputs\n",
    "\n",
    "def test_integration_odog_reform(output_odog_matlab, stimulus):\n",
    "    model = multyscale.models.ODOG_RHS2007(stimulus.shape, visextent)\n",
    "    output2 = model.apply(stimulus)\n",
    "    model.normalize_outputs = normalize_multiple_outputs\n",
    "    output = model.apply(stimulus)\n",
    "    assert np.allclose(output, output2)\n",
    "    #assert np.allclose(output, output_odog_matlab)\n",
    "    \n",
    "test_integration_odog_reform(\"\", stimulus)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "be557be3",
   "metadata": {},
   "source": [
    "This code takes a while to complete, and has a very dissatisfying ending. The formulation right now cannot reproduce traditional output"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
