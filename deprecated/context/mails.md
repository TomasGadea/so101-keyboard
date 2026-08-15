[Robot Learning] First Steps for Your Projects

Gavryushin Alexey

​
Gavryushin Alexey​
​
Mees Oier Andreas;​
Achenbach Liam;​
Bharadwaj Rajiv;​
Guo Huanyu;​
An Tianxu;​+9 others
​
​
​
​
​
​
​
​
​
Dear Students of Robot Learning,

You should have received an invitation by email to a Slack workspace created to coordinate the Robot Learning projects. Please join the workspace in case you have not done so yet. If you did not receive an invitation, please reach out to us.

In the workspace, you will be added to a channel made specifically for your group. You have received or will very soon receive a description of your project, together with references to works you can consult to get started.

Importantly, we ask that before each Thursday session starting next week, you send us a short update using your group's Slack channel, describing your progress and any issues you are currently facing. This will help us better use the slot allotted to your group and keep track of issues requiring our attention.

Regardless of your project and plan, we ask you to complete the following next steps as an introduction and sanity check to ensure everything works:

1. Collect 20 demonstrations of your project's task or a simplified version of it using your SO-101 arms. Please do the same simple motion and try to eliminate any variation between the demonstrations. Importantly, please use the LeRobot dataset format v3 (https://huggingface.co/docs/lerobot/lerobot-dataset-v3) for ALL demonstrations recorded throughout this course.
2. Perform a simple replay of some of the demonstrations with the objects in the place you had them when you recorded the demonstrations, to verify that the demonstrations were recorded correctly. You do not need to train anything yet for this step.
3. Upload the demonstrations to Brev, pick any policy of your choice (keep it simple, check e.g. the HuggingFace LeRobot repo at https://github.com/huggingface/lerobot for models and example code), and train it in a behavior cloning manner to overfit to those demonstrations.
4. Deploy the trained policy. Please make sure that the background and object positions match those seen in the demonstrations. Check if your robot is able to replicate the motion you recorded. If none of your team members has a GPU to deploy the policy, please try asking other groups or reach out to your TAs on Slack.

Please report any problems in the Slack channel created for your group.

Best,
Robot Learning Team

