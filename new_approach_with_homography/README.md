run manual_pose_probe.py to fill calibration.json

---

verify our finger tip position using "rerun" (pip install rerun-sdk)

we have set the finger tip position correctly (according to the above 3d model) use

`rerun new_calibration/so101_ee_tip_vis_rerun_visualization.urdf` to inspect it and /new_calibration/so101_new_calib_ee_tip.urdf for the robot alongside this homography calibration new_approach_with_homography/calibration_ee_tip.json and adjusting the z-plane in pipeline_config.json

BUT this results in the robot being off on some keys

(this is what we use right now) using the manually set finger tip and new_calibration/so101_new_calib.urdf
which is clearly wrong when visualized in the 3d sim above -> this this works perfectly ... we dont know why this works but the above fails


---

### the pipeline.py

tweak it via parameters in pipeline_config.json

- when running check that the keyboard overlay looks perfect


What it does:

take picture -> to ./new_approach_with_homography/images

run 3d_coordinates/vlm_keyboard_coords.py

run new_approach_with_homography/manual_pose_probe.py

and fill out new_approach_with_homography/calibration.json

comp 3 d -> feed to -> go to target

---

Depends on:

making use of this calibration

new_calibration/

specifically how the end effetor tip was manually tuned: new_calibration/so101_new_calib.urdf   <joint name="gripper_frame_joint" type="fixed">


