from astropy.table import Table, column
import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import pandas as pd
from paramiko import SSHClient
from scp import SCPClient

class ParamikoClient(object):
    
    def __init__(self,hostname='bayonet-08.ics.uci.edu',
                 username='zoo',password='GalaxyZoo2'):
        ssh = SSHClient()
        ssh.load_system_host_keys()
        ssh.connect(hostname,username=username,password=password)
        self.ssh = ssh
        self.scp = SCPClient(ssh.get_transport())
        self.ftp = ssh.open_sftp()
        
    def close(self):
        self.ssh.close()


class ArcData():
    
    def __init__(self,dr8id,band='r',save_directory='save_images'):
        self.dr8id = int(dr8id)
        self.band = band
        self.id_string = str(dr8id)
        self.id_suffix = self.id_string[-3:]
        self.remote_directory = ('/home/zoo/SpArcFiRe/SpArcFiRe/{}/{}/'
                                 '{}'.format(self.band,self.id_suffix,
                                             self.id_string))
        self.band_directory = '{}/{}'.format(save_directory,self.band)
        self.galaxy_directory = '{}/{}/{}'.format(save_directory,self.band,
                                                  self.id_string)
        if os.path.isdir(save_directory) is False:
            os.mkdir(save_directory)
        if os.path.isdir(self.band_directory) is False:
            os.mkdir(self.band_directory)
        if os.path.isdir(self.galaxy_directory) is False:
            os.mkdir(self.galaxy_directory)
            
    def measure_chirality(self,galaxy_file):
        galaxy_row_index = galaxy_file['name'] == self.dr8id
        galaxy_row = galaxy_file[galaxy_row_index]
        self.chirality = galaxy_row['chirality_wtdPangSum'][0]
    
    def download_remote_folder(self,client,overwrite=False):
        file_list = client.ftp.listdir(self.remote_directory)
        for file in file_list:
            remote_file = '{}/{}'.format(self.remote_directory,file)
            local_file = '{}/{}'.format(self.galaxy_directory,file)
            if (os.path.exists(local_file)) is True:
                os.remove(local_file) if overwrite is True else None
            if (os.path.exists(local_file)) is False:
                client.scp.get(remote_file,local_file)

    def download_remote_file(self,client,suffix='B_autoCrop',overwrite=False):
        file = '{}-{}.png'.format(self.id_string,suffix)
        remote_file = '{}/{}'.format(self.remote_directory,file)
        local_file = '{}/{}'.format(self.galaxy_directory,file)
        if (os.path.exists(local_file)) is True:
            os.remove(local_file) if overwrite is True else None
        if (os.path.exists(local_file)) is False:
            client.scp.get(remote_file,local_file)
        self.local_file = local_file
        
    def arc_parameters(self,arc_file,galaxy_file,N_max=8):
        self.measure_chirality(galaxy_file)
        row_mask = arc_file['gxyName'] == self.dr8id
        arcs = arc_file[row_mask].sort_values('alenRank')
        arc_table = Table(arcs.as_matrix(),names=arcs.dtypes.index)
        if len(arc_table) > N_max:
            arc_table = arc_table[:N_max]
        
        if (self.chirality == 'Swise') | (self.chirality == 'S-wise'):
            arc_table['chirality_agreement'] = arc_table['pitch_angle'] >= 0
        else:
            arc_table['chirality_agreement'] = arc_table['pitch_angle'] < 0
        arc_table['pitch_angle_absolute'] = \ 
                                   np.absolute(arc_table['pitch_angle'])
        arc_table['delta_r'] = np.absolute(arc_table['r_start'] 
                                           - arc_table['r_end'])
        return arc_table
    
    def display_image(self,ax,suffix='B_autoCrop',file=None,client=None):
        plt.sca(ax)
        if file is not None:
            _ = self.download_remote_file(client,suffix=suffix)
        image_file = '{}/{}-{}.png'.format(self.galaxy_directory,
                                           self.id_string,suffix)
        image = mpimg.imread(image_file)
        h, w = image.shape
        ax.imshow(image,cmap='gray')
        ax.set_xlim(0,w-1)
        ax.set_ylim(h-1,0)
        ax.set_xticks([])
        ax.set_yticks([])
        self.image_height = h
        self.image_width = w
    
    def draw_arcs(self,ax,arc_file,galaxy_file,label=True,N_max=8,colors=None,
                  **kwargs):
        plt.sca(ax)
        if colors is None:
            colors = ('red','blue','orange','magenta','limegreen','orangered',
                      'purple','cyan','green','yellow','black','white')
        arc_table = self.arc_parameters(arc_file,galaxy_file,N_max)
        N_arcs = len(arc_table)
        for n in range(N_arcs):
            arc_row = arc_table[n]
            theta_start = arc_row['math_initial_theta']
            theta_end = theta_start + arc_row['relative_theta_end']
            psi = arc_row['pitch_angle']*(2*math.pi/360)
            initial_radius = arc_row['math_initial_radius']
            r0 = initial_radius/np.exp(-psi*theta_start)
            thetas = np.linspace(theta_start,theta_end,1000)
            x = r0*(np.cos(thetas))*np.exp(-psi*thetas)#+128
            y = r0*(np.sin(thetas))*np.exp(-psi*thetas)#+128
            x_transformed = x + 128
            y_transformed = -y + 128
            ax.plot(x_transformed,y_transformed,
                     color=colors[n],**kwargs)
            if label is True:
                x_centre, y_centre = x_transformed[500], y_transformed[500]
                #x_centre = x_centre+5 if x_centre < 128 else x_centre-5
                y_centre = y_centre+8 if y_centre < 20 else y_centre-8
                text = plt.text(x_centre,y_centre,'{}'.format(n+1),
                                color=colors[n])
                text.set_weight('heavy')
                text.set_size(30)
                text.set_path_effects([path_effects.Stroke(linewidth=2,
                                                           foreground='white'),
                                       path_effects.Normal()])
    
    
    def draw_ellipse(self,ax,galaxy_file,**kwargs):
        plt.sca(ax)
        galaxy_row_index = galaxy_file['name'] == self.dr8id
        galaxy_row = galaxy_file[galaxy_row_index]
        a = galaxy_row['diskMajAxsLen']/2
        b = galaxy_row['diskMinAxsLen']/2
        h = galaxy_row['inputCenterC']
        w = galaxy_row['inputCenterR']
        theta_major = galaxy_row['diskMajAxsAngleRadians']
        thetas = np.linspace(0,2*math.pi,1000)
        r = (a*b)/np.sqrt((b*np.cos(thetas-theta_major))**2
                          + (a*np.sin(thetas-theta_major))**2)
        x = r*np.cos(thetas)
        y = r*np.sin(thetas)
        plt.plot(x+w,-y+h,**kwargs)