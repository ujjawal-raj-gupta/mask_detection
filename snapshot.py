import os
from datetime import datetime

import cv2

VIOLATIONS_DIRNAME = "violations"
VIOLATIONS_DIR = os.path.join(os.path.dirname(__file__), VIOLATIONS_DIRNAME)
os.makedirs(VIOLATIONS_DIR, exist_ok=True)


def save_violation_snapshot(frame, box, mask_status, temp_status, person_id=None):
	"""Save a cropped image of a flagged person and return its relative path.

	frame: full BGR frame from OpenCV
	box: (startX, startY, endX, endY) bounding box for the person
	mask_status: 'Mask' / 'No Mask'
	temp_status: 'Normal' / 'Fever' or None
	Returns the stored relative path (violations/<file>.jpg) or None when the
	person is not a violation or the crop is empty.
	"""
	is_violation = (mask_status == "No Mask") or (temp_status == "Fever")
	if not is_violation:
		return None

	(startX, startY, endX, endY) = box
	h, w = frame.shape[:2]
	startX, startY = max(0, int(startX)), max(0, int(startY))
	endX, endY = min(w, int(endX)), min(h, int(endY))

	if endX <= startX or endY <= startY:
		return None

	cropped = frame[startY:endY, startX:endX]
	if cropped.size == 0:
		return None

	ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	if mask_status == "No Mask" and temp_status == "Fever":
		reason = "nomask_fever"
	elif mask_status == "No Mask":
		reason = "nomask"
	else:
		reason = "fever"

	filename = f"{ts}_{reason}_p{person_id if person_id is not None else 'x'}.jpg"
	filepath = os.path.join(VIOLATIONS_DIR, filename)
	if not cv2.imwrite(filepath, cropped):
		return None

	return f"{VIOLATIONS_DIRNAME}/{filename}"
