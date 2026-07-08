import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
	def __init__(self, maxDisappeared=30, maxDistance=80):
		self.nextObjectID = 0
		self.objects = {}
		self.disappeared = {}
		self.maxDisappeared = maxDisappeared
		self.maxDistance = maxDistance

	def set_frame_width(self, width):
		self.maxDistance = max(80, int(width * 0.18))

	def register(self, centroid):
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]

	def _centroid(self, rect):
		(startX, startY, endX, endY) = rect
		return (int((startX + endX) / 2.0), int((startY + endY) / 2.0))

	def update(self, rects):
		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
			return {}

		inputCentroids = np.array([self._centroid(rect) for rect in rects], dtype="float32")
		if inputCentroids.ndim == 1:
			inputCentroids = inputCentroids.reshape(1, -1)

		if len(self.objects) == 0:
			for i in range(len(inputCentroids)):
				self.register(tuple(inputCentroids[i].astype(int)))
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = np.array(list(self.objects.values()), dtype="float32")
			if objectCentroids.ndim == 1:
				objectCentroids = objectCentroids.reshape(1, -1)

			D = dist.cdist(objectCentroids, inputCentroids)
			rows = D.min(axis=1).argsort()
			cols = D.argmin(axis=1)[rows]

			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue
				if D[row, col] > self.maxDistance:
					continue

				objectID = objectIDs[row]
				self.objects[objectID] = tuple(inputCentroids[col].astype(int))
				self.disappeared[objectID] = 0
				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(D.shape[0])).difference(usedRows)
			unusedCols = set(range(D.shape[1])).difference(usedCols)

			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1
					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			else:
				for col in unusedCols:
					self.register(tuple(inputCentroids[col].astype(int)))

		if len(self.objects) == 0:
			for i in range(len(inputCentroids)):
				self.register(tuple(inputCentroids[i].astype(int)))
			return {
				objectID: rects[i]
				for i, objectID in enumerate(self.objects.keys())
			}

		tracked = {}
		objectIDs = list(self.objects.keys())
		objectCentroids = np.array(list(self.objects.values()), dtype="float32")
		if objectCentroids.ndim == 1:
			objectCentroids = objectCentroids.reshape(1, -1)
		D = dist.cdist(objectCentroids, inputCentroids)
		usedCols = set()

		for row, objectID in enumerate(objectIDs):
			candidates = D[row].argsort()
			for col in candidates:
				if col in usedCols:
					continue
				if D[row, col] <= self.maxDistance:
					tracked[objectID] = rects[col]
					usedCols.add(col)
					break

		return tracked
