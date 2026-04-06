import numpy as np
from gsd import hoomd

def extract_positions(gsd_file, output_file, frames="all"):
    """
    Load particle positions from a GSD file. 

    Parameters
    ----------
    gsd_file : str
        Path to the GSD file.
    output_file : str
        Path to the output text file.
    frames : str, optional
        Which frames to extract. Default is "all". 
        To extract the last N frames, use format "last:N" (e.g., "last:5").
        Any other string will raise a ValueError.

    Returns
    -------
    N_frames : int
        Number of frames processed.
    """
    # Open gsd file
    traj = hoomd.open(name=gsd_file, mode='r')
    
    if frames == "all":
        frames_to_extract = traj
    elif isinstance(frames, str) and frames.startswith("last:"):
        try:
            n = int(frames.split(":", 1)[1])
            frames_to_extract = traj[-n:]
        except (ValueError, IndexError):
            raise ValueError(f"Invalid format for frames: {frames}. Use 'last:N' where N is an integer.")
    else:
        raise ValueError(f"Invalid frames string: {frames}. Must be 'all' or 'last:N' where N is an integer.")

    N_frames = 0
    # Save file if 
    if output_file is not None:
        with open(output_file, 'w') as f:
        
            # Loop through frames
            for frame in frames_to_extract:
                
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
                
    return N_frames
