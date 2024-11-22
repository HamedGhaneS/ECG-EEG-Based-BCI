"""
Real-time Visual Feedback for ECG R-Peak Detection
------------------------------------------------
Author: Hamed Ghane
Date: November 22, 2024

This script provides visual feedback for R-peak detection through LSL markers.
It displays a window that changes color from blue to red for 300ms upon receiving 
each R-peak marker from the ECG detection system.

Workflow:
1. Creates a PsychoPy window (800x600, blue background)
2. Connects to LSL stream 'ECG_R_Peak_Markers'
3. Monitors for incoming markers
4. Changes window color:
   - Blue: Default state
   - Red: For 300ms when R-peak marker received
   
Required Libraries:
- psychopy: for visual display
- pylsl: for LSL communication
- time, sys: for system operations

Usage:
1. Start the R_Peak detection script first
2. Run this script
3. Press 'q' to quit

Note: Requires active LSL stream named 'ECG_R_Peak_Markers'
"""

from psychopy import visual, core, event
import pylsl
import sys
import time

def log_message(message):
    """Print timestamped log messages"""
    print(f"LOG [{time.strftime('%H:%M:%S')}]: {message}")
    sys.stdout.flush()

def create_lsl_inlet():
    """Create LSL inlet for ECG markers"""
    log_message("Searching for R-peak markers stream...")
    streams = pylsl.resolve_stream('name', 'ECG_R_Peak_Markers')
    inlet = pylsl.StreamInlet(streams[0])
    log_message("LSL inlet created successfully")
    return inlet

def main():
    try:
        # Initialize PsychoPy window (not fullscreen)
        log_message("Creating PsychoPy window...")
        win = visual.Window(
            size=[800, 600],  # Smaller window size
            fullscr=False,
            color='blue',
            monitor='testMonitor',
            units='height'
        )
        
        # Create LSL inlet
        inlet = create_lsl_inlet()
        log_message("Starting marker detection...")
        
        # Create timer for color control
        color_timer = core.Clock()
        showing_red = False
        RED_DURATION = 0.3  # 300ms
        
        while True:
            # Check for quit
            if 'q' in event.getKeys():
                log_message("Quit signal received")
                break
            
            try:
                # Check for R-peak marker (non-blocking)
                sample, timestamp = inlet.pull_sample(timeout=0.0)
                
                if sample is not None:
                    # Marker received - change to red
                    win.color = 'red'
                    showing_red = True
                    color_timer.reset()  # Start timing
                    log_message("Marker received - Screen Red")
                
                # Check if it's time to return to blue
                if showing_red and color_timer.getTime() >= RED_DURATION:
                    win.color = 'blue'
                    showing_red = False
                    log_message("Returning to Blue")
                
                # Update window
                win.flip()
                
            except Exception as e:
                log_message(f"Error in marker processing: {str(e)}")
                break
            
            # Small wait to prevent CPU overload
            core.wait(0.001)
    
    except Exception as e:
        log_message(f"Error: {str(e)}")
    
    finally:
        log_message("Cleaning up...")
        win.close()
        core.quit()

if __name__ == "__main__":
    main()
