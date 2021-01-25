import math
import os
import sys
import numpy as np
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

class ComplexNumber:
    def __init__(self, x, y, index, val):
        self.x = x
        self.y = y
        self.index = index
        self.val = val

    def get_index(self):
        return self.index

    def get_val(self):
        return self.val

class AnomalousMovingMNIST:
    '''
    Args:
    anom_frames: list of anomalous frame indexes that must appear in the sequence
    num_anoms_per_frame: how many MNIST numbers have to be anomalous in an anomalous frame
    n_clusters: number of clusters to perform K-Means.
        More clusters imply smaller cluster size, more similarity between instances but less variety.
        Less clusters imply bigger cluster size, less similarity between instances but more variety.
    '''
    def __init__(self,
                 anom_frames,
                 num_anoms_per_frame,
                 num_sequences=20000,
                 shape=(64, 64),
                 num_frames=30,
                 original_size=28,
                 nums_per_image=2,
                 path_data = '',
                 path_labels = '',
                 path_tSNE = '',
                 n_clusters = 9,
                 dest='anomovingmnistdata',):

        self.shape = shape
        self.num_frames = num_frames
        self.num_sequences = num_sequences
        self.original_size = original_size
        self.nums_per_image = nums_per_image
        self.dest = dest
        self.path_data = path_data
        self.path_labels = path_labels
        self.path_tSNE = path_tSNE
        self.n_clusters = n_clusters
        self.anom_frames = anom_frames
        self.num_anoms_per_frame = num_anoms_per_frame

    def get_array_from_image(self, im, mean=0, std=1):
        '''
        Args:
            im: Image
            shift: Mean to subtract
            std: Standard Deviation to subtract

        Returns:
            Image in np.float32 format, in width height channel format. With values in range 0,1
            Shift means subtract by certain value. Could be used for mean subtraction.
        '''
        width, height = im.size
        arr = im.getdata()
        c = int(np.product(arr.size) / (width * height))

        return (np.asarray(arr, dtype=np.float32).reshape((height, width, c)).transpose(2, 1, 0)) / std

    def get_image_from_array(self, X, index, mean=0, std=1):
        '''
        Args:
            X: Dataset of shape N x C x W x H
            index: Index of image we want to fetch
            mean: Mean to add
            std: Standard Deviation to add
        Returns:
            Image with dimensions H x W x C or H x W if it's a single channel image
        '''
        ch, w, h = X.shape[1], X.shape[2], X.shape[3]
        ret = (((X[index] + mean)) * std).reshape(ch, w, h).transpose(2, 1, 0)#.clip(0, 255).astype(np.uint8)
        if ch == 1:
            ret = ret.reshape(h, w)
        return ret

    #Load dataset from sklearn
    def load_dataset_from_sklearn(self):
        data = []
        labels = []
        data_reshaped = []
        try:
            data = np.load(self.path_data, allow_pickle=True)
            labels = np.load(self.path_labels, allow_pickle=True)
            print('Loaded dataset from file system')
        except:
            from sklearn.datasets import fetch_openml
            print("Downloading MNIST dataset from sklearn")
            data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
            data = np.asarray(data)
            labels = np.asarray(labels)
            print(data)
            print(data.shape)
            print(labels)
            print(labels.shape)
            data_reshaped = data.reshape(-1, 1, 28, 28).transpose(0, 1, 3, 2)
            np.save('data', data)
            np.save('labels', labels)
            print(("done"))

        return data[-10000:], data_reshaped[-10000:] / np.float32(255), labels[-10000:]

    def perform_tSNE(self, data):
        img_embed = None
        try:
            img_embed = np.load(self.path_tSNE)
            print("Loading existing t-SNE embedding from file system")
            print("done")
        except:
            print("Performing t-SNE on a dataset of shape "+str(data.shape)+"...")
            tsne = TSNE(n_components=3)
            img_embed = tsne.fit_transform(data)
            np.save(self.path_tSNE, img_embed)
        return img_embed

    def perform_Birch(self, embedding, labels, n_clusters):
        print("Performing Birch. K =",n_clusters)
        clusters_dimension = [0]*n_clusters
        birch = MiniBatchKMeans(n_clusters=n_clusters).fit(embedding)
        distances = MiniBatchKMeans(n_clusters=n_clusters).fit_transform(embedding)
        for i in birch.labels_:
                clusters_dimension[i]+=1
        clusters = [ [] for i in range(n_clusters)]
        pos = 0
        for i in birch.labels_:
                clusters[i].append(ComplexNumber(embedding[pos, 0], embedding[pos, 1], pos, labels[pos]))
                pos+=1

        med = np.argmin(distances, axis=0)
        clusters_others = {}
        for i in range(len(clusters)):
            cluster = clusters[i]
            medoid = med[i]
            others = [x.get_index() for x in cluster if x.get_val()!=str(labels[medoid])]
            clusters_others[medoid]=others

        return clusters, clusters_others, birch.labels_

    def generate_moving_mnist(self,
                              shape=(64, 64),
                              num_frames=20,
                              num_sequences=10000,
                              original_size=28,
                              nums_per_image=2):
        '''
        Args:
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in a particular movement/animation/gif
        num_sequences: Number of movement/animations/gif to generate
        original_size: Real size of the images (eg: MNIST is 28x28)
        nums_per_image: Digits per movement/animation/gif.

        Returns:
        Dataset of np.uint8 type with dimensions num_frames * num_sequences x 1 x new_width x new_height
        '''

        print("Generating dataset of shape ("+str(self.num_sequences)+", "+ str(self.num_frames)+", 1, "+str(self.shape[0])+", "+str(self.shape[1])+")")

        data, mnist, labels = self.load_dataset_from_sklearn()

        img_embed = self.perform_tSNE(data)
        _, med_other, predicted_labels = self.perform_Birch(img_embed, labels, n_clusters=self.n_clusters)
        np.save("predicted_labels", predicted_labels)

        '''
        index_list = medoids
        index_list_false = candidates
        mnist_false = data[index_list_false].reshape(-1, 1, 28, 28)#.transpose(0, 1, 3, 2)
        '''
        mnist = data.reshape(-1, 1, 28, 28)#.transpose(0, 1, 3, 2)

        print("Building dataset...")
        width, height = self.shape
        # Get how many pixels can we move around a single image
        lims = (x_lim, y_lim) = width - original_size, height - original_size

        # Create a dataset of shape of num_frames * num_sequences x 1 x new_width x new_height
        # Eg : 3000000 x 1 x 64 x 64
        dataset = np.empty((self.num_frames * self.num_sequences, 1, width, height), dtype=np.uint8)

        for img_idx in range(self.num_sequences):
            direcs = np.pi * (np.random.rand(self.nums_per_image) * 2 - 1)
            speeds = np.random.randint(5, size=self.nums_per_image) + 2
            veloc = np.asarray(
                [(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])

            # Get a list containing two PIL images randomly sampled from the database
            #casual_index = np.random.randint(0, mnist.shape[0], nums_per_image)
            casual_index = np.random.randint(0, self.n_clusters, self.nums_per_image)
            mnist_images = []
            image_false = None
            mnist_images_false = None
            medoids = list(med_other.keys())

            for r in casual_index:
                mnist_images.append(Image.fromarray(self.get_image_from_array(mnist, medoids[r], mean=0)).resize((original_size, original_size), Image.ANTIALIAS))

            anom_list_med_1 = med_other[medoids[casual_index[0]]]
            anom_list_med_2 = med_other[medoids[casual_index[1]]]
            np.random.shuffle(anom_list_med_1)
            np.random.shuffle(anom_list_med_2)

            anom_index_1 = medoids[casual_index[0]]
            while labels[anom_index_1] == labels[medoids[casual_index[0]]]:
                anom_index_1 = np.random.choice(anom_list_med_1)

            anom_index_2 = medoids[casual_index[1]]
            while labels[anom_index_2] == labels[medoids[casual_index[1]]]:
                anom_index_2 = np.random.choice(anom_list_med_2)

            image_false1 = Image.fromarray(self.get_image_from_array(mnist, anom_index_1, mean=0)).resize((original_size, original_size), Image.ANTIALIAS)
            image_false2 = Image.fromarray(self.get_image_from_array(mnist, anom_index_2, mean=0)).resize((original_size, original_size), Image.ANTIALIAS)

            if(self.num_anoms_per_frame == 1):
                rand_idx = np.random.randint(2)
                mnist_images_false = [image_false1, image_false2]
                mnist_images_false[rand_idx] = mnist_images[rand_idx]

            if(self.num_anoms_per_frame == 2):
                mnist_images_false = [image_false1, image_false2]

            # Generate tuples of (x,y) i.e initial positions for nums_per_image (default : 2)
            positions = np.asarray(
                [(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(self.nums_per_image)])
            positions_false = np.asarray(
                [(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(self.nums_per_image)])

            # Generate new frames for the entire num_framesgth
            for frame_idx in range(self.num_frames):
                canvases = [Image.new('L', (width, height)) for _ in range(self.nums_per_image)]
                canvas = np.zeros((1, width, height), dtype=np.float32)

                # In canv (i.e Image object) place the image at the respective positions
                # Super impose both images on the canvas (i.e empty np array)
                for i, canv in enumerate(canvases):
                    im = mnist_images[i]
                    if(frame_idx in self.anom_frames):
                        im = mnist_images_false[i]
                        #canv.paste(im, tuple(positions_false[i].astype(int)))
                    #else:
                    canv.paste(im, tuple(positions[i].astype(int)))
                    canvas += self.get_array_from_image(canv, mean=0)

                # Get the next position by adding velocity
                next_pos = positions + veloc

                # Iterate over velocity and see if we hit the wall
                # If we do then change the  (change direction)
                for i, pos in enumerate(next_pos):
                    for j, coord in enumerate(pos):
                        if coord < -2 or coord > lims[j] + 2:
                            veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

                # Make the permanent change to position by adding updated velocity
                positions = positions + veloc

                # Add the canvas to the dataset array
                dataset[img_idx * num_frames + frame_idx] = ((canvas).clip(0, 255)).astype(np.float32)
        #Reshape dataset to have a shape like (30, 100, 1, 64, 64)
        dataset = dataset.reshape(self.num_sequences, self.num_frames, 1, width, height)
        print("done! :)")
        return dataset

    def generate_ano_mnist(self):
        data = self.generate_moving_mnist()

        n = self.num_sequences * self.num_frames
        np.savez(self.dest, anommnist=data)


def create_gifs(dat, number_of_gifs=20):
    for j in range(0, number_of_gifs):
        fig = plt.figure()
        ims = []
        for i in range(0,20):
            im = plt.imshow(dat[j,i,0,])
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True, repeat_delay=10000)
        ani.save('dynamic_images' + str(j) + '.gif')
    return None

def plot_3d_sne():
    tsne_embedding = np.load("sne.npy")
    predicted_labels = np.load("predicted_labels.npy", allow_pickle=True)

    xs = list(map(lambda x: x[0], tsne_embedding))
    ys = list(map(lambda x: x[1], tsne_embedding))
    zs = list(map(lambda x: x[2], tsne_embedding))

    fig = plt.figure(num=None, figsize=(18, 12), dpi=80, facecolor='w', edgecolor='k')
    N = 15
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c=predicted_labels, cmap=cmap)

    plt.show()
    return None

def compute_and_print_elbow_method_for_birch():
    img_embed = np.load("sne.npy")
    inertias = []
    range_n_clusters = range(2, 30)
    mapping1 = {}

    for n_clusters in range_n_clusters:
        mbkmeansmodel = MiniBatchKMeans(n_clusters=n_clusters).fit(img_embed)

        inertias.append(mbkmeansmodel.inertia_)

        mapping1[n_clusters] = sum(np.min(cdist(img_embed, mbkmeansmodel.cluster_centers_,'euclidean'),axis=1)) / img_embed.shape[0]

    for key,val in mapping1.items():
        print(str(key)+' : '+str(val))

    plt.plot(range_n_clusters, inertias, 'bx-')
    plt.xlabel('Values of n_clusters')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()

    return None

def main():
    gen = AnomalousMovingMNIST(
        anom_frames=[5],
        num_anoms_per_frame=1,
        num_frames=20,
        num_sequences=200,
        n_clusters=15,
        path_data='data.npy',
        path_labels='labels.npy',
        path_tSNE='sne.npy',
        dest='anommnist')

    gen.generate_ano_mnist()
    dat = np.load('anommnist.npz')['anommnist']

    create_gifs(dat)
    plot_3d_sne()

if __name__ == "__main__":
    main()
