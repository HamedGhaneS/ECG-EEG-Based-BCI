from psychopy import core, sound
import numpy as np
try:
    import pyxid2 as pyxid  # Try newer version first
except ImportError:
    import pyxid    # Fall back to older version if needed

def test_cedrus():
    # First, let's create our test tone
    # Using a 500Hz tone that lasts 0.2 seconds
    beep = sound.Sound(value=500, secs=0.2)
    
    # Initialize the Cedrus box with multiple attempts
    # Sometimes the connection isn't successful on the first try
    print("Looking for Cedrus device...")
    cedrus_box = None
    
    for attempt in range(10):
        try:
            devices = pyxid.get_xid_devices()
            core.wait(0.1)  # Short pause between attempts
            if devices:
                cedrus_box = devices[0]  # Get the first connected device
                cedrus_box.clock = core.Clock()  # Add a clock for timing
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed. Retrying...")
            continue
    
    if not cedrus_box:
        print("Could not find Cedrus device. Please check connection.")
        core.quit()
    
    print("Cedrus device found!")
    print("Press any button on the response box (press 'q' on keyboard to quit)")
    
    # Main loop to check for button presses
    while True:
        # Check for keyboard quit command
        if event.getKeys(['q']):
            print("Quit command received. Ending test.")
            break
        
        # Poll the response box for any button press
        cedrus_box.poll_for_response()
        while len(cedrus_box.response_queue):
            response = cedrus_box.get_next_response()
            
            if response['pressed']:  # Only react to button press, not release
                # Print which button was pressed
                print(f"Button {response['key']} was pressed!")
                
                # Play the beep
                beep.play()
                
                # Wait for beep to finish
                core.wait(0.2)
            
            # Clear any remaining responses
            cedrus_box.poll_for_response()
        
        # Small wait to prevent CPU overload
        core.wait(0.001)
    
    print("Test complete!")

if __name__ == "__main__":
    test_cedrus()