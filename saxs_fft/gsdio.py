import numpy as np
from gsd import hoomd

def extract_positions(gsd_file, output_file):
    """
    Load particle positions from a GSD file. 

    Parameters
    ----------
    gsd_file : str
        Path to the GSD file.
    out_file : str
        Path to the output text file.
    Returns
    -------
    positions : np.ndarray
        Particle positions. Shape:
        - (N, D) if single frame
        - (F, N, D) if all frames
    """
    # Open gsd file
    traj = hoomd.open(name=gsd_file, mode='r')
    N_frames = 0
    # Save file if 
    if output_file is not None:
        with open(output_file, 'w') as f:
        
            # Loop through frames
            for frame in traj:
                
                # Read data
                N = frame.particles.N
                box = frame.configuration.box
                x = frame.particles.position
                
                # Write to text file
                f.write('{:d}\n'.format(N))
                f.write('{:.5f} {:.5f} {:.5f}\n'.format(box[0], box[1], box[2]))
                for pos in x:
                    f.write('{:.5f} {:.5f} {:.5f}\n'.format(pos[0], pos[1], pos[2]))
                f.write('\n')
                N_frames += 1
                
    print(f'Total frames extracted: {N_frames}')