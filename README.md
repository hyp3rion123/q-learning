This code demonstrates communication in a two-agent reinforcement learning problem using q-learning. One agent, the receiver, is in a 5 × 5 grid-world (with walls) and starts at the
center location. A prize is hidden (from the receiver) somewhere on the grid. The prize is placed randomly but cannot be in a wall location, and is never placed at the receiver’s starting position. The second agent, the sender, knows the location of the prize only, and can send the receiver a single message as one of a set of N arbitrary symbols. Thus, since the symbols have no “meaning” at the start, the agents have to jointly establish this meaning for each of the N symbols. For example, if N = 25, the sender can precisely specify where the prize is, but the two agents still have to assign meaning to the symbols. If N < 2, the sender is useless. The sender and the receiver each get the prize (a reward of 1.0 each) upon the receiver entering the cell that contains the prize. After receiving the sender’s message, the receiver acts until it finds the reward or is terminated.

This problem is solved using a two-agent q-learning model. The sender agent’s states are the location of the prize on the grid. Its actions are the possible symbols (messages) it can send. The receiver agent’s states are tuples giving the received message and the current location (which is fully observable). They will be tested within three environments described below:

![image](https://github.com/hyp3rion123/q-learning/assets/78772133/45375b0d-efdf-4847-a617-9ba0860d69ef)
