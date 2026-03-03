import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from PIL import Image
import matplotlib.image as mpimg
import matplotlib as mpl

char_path = '/workspace/projects/motif/chars'

def make_directory(path, foldername, verbose=1):
	"""make a directory"""

	if not os.path.isdir(path):
		os.mkdir(path)
		print("making directory: " + path)

	outdir = os.path.join(path, foldername)
	if not os.path.isdir(outdir):
		os.mkdir(outdir)
		print("making directory: " + outdir)
	return outdir


def normalize_pwm(pwm, factor=None, max=None):

	if not max:
		max = np.max(np.abs(pwm))
	pwm = pwm/max
	if factor:
		pwm = np.exp(pwm*factor)
	norm = np.outer(np.ones(pwm.shape[0]), np.sum(np.abs(pwm), axis=0))
	return pwm/norm


def get_images_of_motifs(core_motifs, core_names, savepath_results):

    i=0
    for j in range(0,len(core_motifs),2):

        save_path = make_directory(savepath_results,'ground_truth_motifs')
        fig = plt.figure(figsize = (3,1))
        plt.subplot(1,2,1)
        logo = seq_logo(pwm=core_motifs[j], height=100, nt_width=50, norm=0, alphabet='dna')
        plot_seq_logo(logo, nt_width=20, step_multiple=4)    
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(core_names[i], fontsize=14)

        plt.subplot(1,2,2)
        logo = seq_logo(pwm=core_motifs[j+1], height=100, nt_width=50, norm=0, alphabet='dna')
        plot_seq_logo(logo, nt_width=20, step_multiple=4)    
        plt.xticks([])
        plt.yticks([])

        fig.savefig(os.path.join(save_path, core_names[i]+'.pdf'), format='pdf', dpi=200, bbox_inches='tight')
        
        i += 1


def plot_seq_logo(logo, nt_width=None, step_multiple=None):
	plt.imshow(logo, interpolation='none')
	if nt_width:
		num_nt = logo.shape[1]/nt_width
		if step_multiple:
			step_size = int(num_nt/(step_multiple+1))
			nt_range = range(step_size, step_size*step_multiple)
			plt.xticks([step_size*nt_width, step_size*2*nt_width, step_size*3*nt_width, step_size*4*nt_width],
						[str(step_size), str(step_size*2), str(step_size*3), str(step_size*4)])
		else:
			plt.xticks([])
		#plt.yticks([0, 50], ['2.0','0.0'])
		ax = plt.gca()
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.yaxis.set_ticks_position('none')
		ax.xaxis.set_ticks_position('none')
	else:
		plt.imshow(logo, interpolation='none')
		plt.axis('off');


def load_alphabet(alphabet, colormap='standard'):

	def load_char(char, color):
		colors = {}
		colors['green'] = [10, 151, 21]
		colors['red'] = [204, 0, 0]
		colors['orange'] = [255, 153, 51]
		colors['blue'] = [0, 0, 204]
		colors['cyan'] = [153, 204, 255]
		colors['purple'] = [178, 102, 255]
		colors['grey'] = [160, 160, 160]
		colors['black'] = [0, 0, 0]

		img = mpimg.imread(os.path.join(char_path, char+'.eps'))
		img = np.mean(img, axis=2)
		x_index, y_index = np.where(img != 255)
		y = np.ones((img.shape[0], img.shape[1], 3))*255
		for i in range(3):
			y[x_index, y_index, i] = colors[color][i]
		return y.astype(np.uint8)


	colors = ['green', 'blue', 'orange', 'red']
	if alphabet == 'dna':
		letters = 'ACGT'
		if colormap == 'standard':
			colors = ['green', 'blue', 'orange', 'red']
		chars = []
		for i, char in enumerate(letters):
			chars.append(load_char(char, colors[i]))

	elif alphabet == 'rna':
		letters = 'ACGU'
		if colormap == 'standard':
			colors = ['green', 'blue', 'orange', 'red']
		chars = []
		for i, char in enumerate(letters):
			chars.append(load_char(char, colors[i]))


	elif alphabet == 'structure': # structural profile

		letters = 'PHIME'
		if colormap == 'standard':
			colors = ['blue', 'green', 'orange', 'red', 'cyan']
		chars = []
		for i, char in enumerate(letters):
			chars.append(load_char(char, colors[i]))

	elif alphabet == 'pu': # structural profile

		letters = 'PU'
		if colormap == 'standard':
			colors = ['cyan', 'purple']
		elif colormap == 'bw':
			colors = ['black', 'grey']
		chars = []
		for i, char in enumerate(letters):
			chars.append(load_char(char, colors[i]))

	return chars



def seq_logo(pwm, height=30, nt_width=10, norm=0, alphabet='dna', colormap='standard'):

	def get_nt_height(pwm, height, norm):

		def entropy(p):
			s = 0
			for i in range(len(p)):
				if p[i] > 0:
					s -= p[i]*np.log2(p[i])
			return s

		num_nt, num_seq = pwm.shape
		heights = np.zeros((num_nt,num_seq));
		for i in range(num_seq):
			if norm == 1:
				total_height = height
			else:
				total_height = (np.log2(num_nt) - entropy(pwm[:, i]))*height;
			if alphabet == 'pu':
				heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height));
			else:
				heights[:,i] = np.floor(pwm[:,i]*np.minimum(total_height, height*2));

		return heights.astype(int)


	# get the alphabet images of each nucleotide
	chars = load_alphabet(alphabet, colormap)

	# get the heights of each nucleotide
	heights = get_nt_height(pwm, height, norm)

	# resize nucleotide images for each base of sequence and stack
	num_nt, num_seq = pwm.shape
	width = np.ceil(nt_width*num_seq).astype(int)

	if alphabet == 'pu':
		max_height = height
	else:
		max_height = height*2
	#total_height = np.sum(heights,axis=0) # np.minimum(np.sum(heights,axis=0), max_height)
	logo = np.ones((max_height, width, 3)).astype(int)*255;
	for i in range(num_seq):
		nt_height = np.sort(heights[:,i]);
		index = np.argsort(heights[:,i])
		remaining_height = np.sum(heights[:,i]);
		offset = max_height-remaining_height

		for j in range(num_nt):
			if nt_height[j] > 0:
				# resized dimensions of image
				char_img = Image.fromarray(chars[index[j]].astype('uint8'))
				nt_img = np.array(char_img.resize((nt_width, nt_height[j]), resample=Image.BICUBIC))

				# determine location of image
				height_range = range(remaining_height-nt_height[j], remaining_height)
				width_range = range(i*nt_width, i*nt_width+nt_width)

				# 'annoying' way to broadcast resized nucleotide image
				if height_range:
					for k in range(3):
						for m in range(len(width_range)):
							logo[height_range+offset, width_range[m],k] = nt_img[:,m,k];

				remaining_height -= nt_height[j]

	return logo.astype(np.uint8)




def plot_filter_logos(W, figsize=(10, 7), height=25, nt_width=10, norm=0, 
                      alphabet='dna', norm_factor=3, num_rows=None, save_path=None):
    """
    Plots a grid of sequence logos for CNN filters.
    W shape expected: [num_filters, window_size, 4]
    """
    # W is already [num_filters, window_size, 4] from our PyTorch function
    num_filters = W.shape[0]
    
    # Calculate grid dimensions
    if not num_rows:
        num_rows = int(np.ceil(np.sqrt(num_filters)))
        num_cols = num_rows
    else:
        num_cols = int(np.ceil(num_filters / num_rows))
        if num_filters % num_rows != 0:
            num_cols += 1

    fig = plt.figure(figsize=figsize)
    grid = mpl.gridspec.GridSpec(num_rows, num_cols)
    # Adjust spacing to make it readable
    grid.update(wspace=0.2, hspace=0.2, left=0.1, right=0.2, bottom=0.1, top=0.2)

    MAX = np.max(W) if norm else None

    for i in range(num_filters):
        if i >= num_rows * num_cols:
            break
            
        ax = plt.subplot(grid[i])
        
        # Apply normalization logic
        if norm_factor:
            # Reusing your normalize_pwm logic
            W_norm = normalize_pwm(W[i], factor=norm_factor, max=MAX)
        else:
            W_norm = W[i]
            
        # Generate the logo array using your existing seq_logo
        # Note: W_norm is [window_size, 4], seq_logo usually expects [4, window_size]
        # so we transpose W_norm.T
        logo = seq_logo(W_norm.T, height=height, nt_width=nt_width, norm=0, alphabet=alphabet)
        
        # Plot using your existing plot_seq_logo
        plot_seq_logo(logo, nt_width=nt_width, step_multiple=None)
        
        ax.set_title(f'Filter {i}', fontsize=8)
        ax.set_yticks([])
        ax.set_xticks([]) # Clean up x-axis as well for better grid view

    fig.set_size_inches(100, 100)
    fig.savefig(save_path, format='pdf', dpi=200, bbox_inches='tight')
    plt.close()
    