"""
Cedrus Response Box Testing Script
Author: Hamed Ghane
Date: December 12, 2024

This script provides functionality to test a Cedrus response box by:
1. Establishing connection with the device
2. Monitoring button presses and maintaining their history
3. Providing audio feedback (beep) for button presses
4. Displaying running history of all button presses
5. Automatically cleaning up the connection after use

Key Variables and Methods Explanation:
1. Device-Related:
   - cedrus_box: Main device object (XID device instance)
   - cedrus_box.product_id: Device model identifier (XID built-in property)
   - cedrus_box.response_queue: Buffer for button responses (XID built-in property)
   - cedrus_box.clock: Timer for response timing (PsychoPy Clock object)

2. Device Methods:
   - poll_for_response(): XID built-in method to check for new responses
   - clear_response_queue(): XID built-in method to empty response buffer
   - reset_rt_timer(): XID built-in method to reset reaction time clock
   - get_next_response(): XID built-in method to get next response from queue

3. Response Dictionary Keys (returned by get_next_response()):
   - response['pressed']: Boolean for press/release (XID built-in)
   - response['key']: Integer for button number (XID built-in)
   - response['port']: Integer for device port (XID built-in)
   - response['time']: Response timestamp (XID built-in)

4. Custom Variables:
   - button_history: List storing press messages (custom implementation)
   - start_time: Test start timestamp (built-in time.time())
   - max_duration: Test duration in seconds (user parameter)
   - devices: List of connected XID devices (from pyxid.get_xid_devices())

Dependencies:
- psychopy: For timing and sound
- pyxid2/pyxid: For Cedrus device communication
- numpy: For numerical operations
- IPython: For Jupyter display functionality
"""

# Import required libraries
from psychopy import core, sound  # For timing control and audio feedback
import numpy as np              # For numerical operations (if needed)
import time                     # For timing and delays
from IPython import display     # For Jupyter notebook display clearing

# Try to import the newer version of pyxid first, fall back to older if not available
try:
    import pyxid2 as pyxid     # Newer version of Cedrus device interface
except ImportError:
    import pyxid               # Older version as fallback

def clear_xid_devices():
    """
    Clear and reset all XID device connections.
    
    Variables and Methods Used:
    - pyxid.get_xid_devices(): XID built-in function to list connected devices
    - device.clear_response_queue(): XID built-in method to clear response buffer
    - time.sleep(): Python built-in function for delays
    
    Returns:
    None
    """
    try:
        # Get list of all connected devices
        devices = pyxid.get_xid_devices()
        # Attempt to clear each device's queue
        for device in devices:
            try:
                device.clear_response_queue()  # Clear pending responses
                time.sleep(0.1)                # Short delay for processing
            except:
                pass  # Silently handle any device-specific errors
    except:
        pass  # Silently handle any general errors
    time.sleep(0.5)  # Final delay to ensure USB port release

def test_cedrus(max_duration=30):
    """
    Test Cedrus response box functionality with time limit.
    
    Key Variables and Methods:
    1. Device Connection:
       - cedrus_box: XID device instance
       - devices: List from pyxid.get_xid_devices()
    
    2. Sound:
       - beep: PsychoPy Sound object (500Hz, 0.2s)
    
    3. Button History:
       - button_history: List of press messages
       - press_msg: Formatted string of button press
    
    4. Timing:
       - start_time: From time.time()
       - max_duration: Test duration parameter
    
    Parameters:
    max_duration : int
        Maximum test duration in seconds (default: 30)
        
    Returns:
    None
    """
    try:
        # Initial cleanup of any existing connections using XID methods
        clear_xid_devices()
        
        # Create PsychoPy Sound object for audio feedback
        # value=500: 500Hz tone frequency
        # secs=0.2: 200ms duration
        beep = sound.Sound(value=500, secs=0.2)
        
        # Initialize device connection variables
        print("Looking for Cedrus device...")
        cedrus_box = None  # Will hold XID device instance
        
        # Initialize custom list for tracking button presses
        button_history = []
        
        # Device connection attempts loop
        for attempt in range(10):  # attempt: loop counter (built-in range)
            try:
                # Get available devices using XID built-in function
                devices = pyxid.get_xid_devices()
                if devices:
                    # Store first available device
                    cedrus_box = devices[0]
                    try:
                        # Initialize device using XID built-in methods
                        cedrus_box.reset_rt_timer()        # Reset reaction time clock
                        cedrus_box.clear_response_queue()  # Clear response buffer
                    except AttributeError:
                        pass
                    time.sleep(0.1)
                    print(f"Attempt {attempt + 1}: Device found")
                    break
                else:
                    print(f"Attempt {attempt + 1}: No devices found")
                    time.sleep(0.5)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(0.5)
                continue
        
        if not cedrus_box:
            print("\nCould not find Cedrus device. Please try these steps:")
            print("1. Unplug the device")
            print("2. Wait 5 seconds")
            print("3. Plug it back in")
            print("4. Wait 5 seconds")
            print("5. Run the script again")
            return
        
        print("\nCedrus device found!")
        print(f"Device info: {cedrus_box.product_id}")
        print(f"\nTest will run for {max_duration} seconds.")
        print("Press any button on the response box to test.")
        
        # Main testing loop
        start_time = time.time()  # Record start time (built-in function)
        while (time.time() - start_time) < max_duration:
            # Check for responses using XID built-in method
            cedrus_box.poll_for_response()
            while len(cedrus_box.response_queue):  # Check XID response buffer
                # Get response dictionary using XID built-in method
                response = cedrus_box.get_next_response()
                
                if response['pressed']:  # XID response property
                    # Create and store button press message
                    press_msg = f"Button {response['key']} was pressed!"  # response['key']: XID button identifier
                    button_history.append(press_msg)
                    beep.play()          # PsychoPy Sound method
                    core.wait(0.2)       # PsychoPy timing function
                
                # Clear additional responses using XID method
                cedrus_box.poll_for_response()
            
            core.wait(0.001)  # PsychoPy timing function
            # Update Jupyter display using IPython method
            display.clear_output(wait=True)
            
            # Display countdown and instructions
            print(f"\nRunning... {max_duration - int(time.time() - start_time)} seconds remaining")
            print("Press any button on the response box to test.")
            print("\nButton Press History:")
            
            # Display complete button press history
            for msg in button_history:
                print(msg)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    finally:
        # Cleanup using XID methods
        try:
            if 'cedrus_box' in locals():
                cedrus_box.clear_response_queue()  # XID method
            clear_xid_devices()  # Custom function using XID methods
            print("\nCleanup completed. Device connection closed.")
                        
        except:
            pass

# Script entry point check (Python built-in variable)
if __name__ == "__main__":
    test_cedrus()