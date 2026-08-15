Please use the LeRobotDataset v3.0 format to record any robotic demonstrations.

Dear Keyboard Task Groups,

Here are the details for project 4. Please pay attention to the details of the evaluation and rules.

We do not put any restrictions on the model architecture/training procedure used for your feature extraction and policy. We allow architectures trained from scratch, as well as any pre-trained, out-of-the-box policy/feature extractor, or post-training method to maximize success rates. Your training data can be sourced from publicly available resources or generated through teleoperation or synthetic means.

As mentioned before, please do not use any custom hardware except for what was provided to you in the box you received from the course. Also, please treat the hardware with care, as it will need to be returned to us after the completion of the course. Importantly, we do allow the use of a pen or similar tool to type on the keyboard, similar to the image shown on the lecture slides. During evaluation, the robot may start with the tool in its gripper or pick it up from any position in the workspace.

Evaluation:

For the main evaluation, the environment will be standardized: The robot will be placed on a light gray table (approximately #B8ADA9), and the paper sheet or keyboard (black keyboard with US layout, which we will provide to you) will be placed in front of it.

The project goal is defined by the following milestones, with a maximum of 150/150 points available for the main evaluation:
Eval 1 (50 pts): Reaching predefined points.
This subtask can be completed in two ways:
(1) Success is defined as sequentially moving the gripper to four differently colored points (#FF0000, #00FF00, #0000FF, #7FFFFF in that order) on an A4 paper sheet, each ~1.5cm in diameter. The positions of the points will be randomized over the sheet. 12.5 points will be given for each reached point (gripper hovers at most 3cm above the point and overlaps at least 1/2 of it), for as long as the sequence is followed correctly and in-order. The task must be completed in 25 seconds.
(2) Success is defined as pressing the space, enter, R and L keys sequentially and in that order on a keyboard. 12.5 points will be given for each pressed key, for as long as the sequence is followed correctly and in-order. You may provide this sequence as input to your policy if you wish. The task must be completed in 40 seconds.
Eval 2 (50 pts): Press defined keys on a laptop keyboard. Your policy will receive any character in the latin alphabet (a-z) as input and must press the correct key within 10 seconds. 16 rollouts will be conducted in total per group, with each rollout giving 3.125 points if the right key is pressed in time. The exact sequence of keys will not be made known in advance, but will be identical across groups.
Eval 3 (50 pts): Type words on an unseen keyboard. Your policy will receive an arbitrary sentence (a-z, whitespace) of 2 to 4 words as input. Per rollout, you will receive max(0, 5 - d) points after 10 * len(s) seconds, where d is the Levenshtein distance of the resulting character sequence to the input sentence and len(s) is the length of the input sentence, counting spaces. You may assume len(s) <= 15. We will perform 10 rollouts per group, each with a different sentence. The sentences will not be made known in advance, but will be identical across groups. Further, the keyboard we will use will be different, but you may assume the same keyboard layout and color.
Bonus (50pts): Fastest spelling of words on the keyboard. 
For this evaluation, we will calculate for each group their policy’s total duration Ti for typing all sequences in eval 3. For any sequence with errors, i.e. Levenshtein distance > 0 of the typed sequence to the input sequence, we will add the maximum possible time (i.e. 10 * len(s)) for that sequence to Ti; if the sequence was typed correctly in t seconds, then t will be added to Ti. We will then sort all groups by their Ti. The group with the lowest Ti will receive 50 bonus points, the second-lowest will receive 40 points, etc. Starting from the 6th-lowest Ti, no bonus points will be distributed.

Best regards,
Project 4 TAs


References:


https://code-as-policies.github.io/
