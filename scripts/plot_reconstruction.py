import sys
sys.path.append('/home/lukas/Pangolin/build/src')
sys.path.append('/home/pangolin/build/src') # for inside docker container
sys.path.append("..")

import os
import pickle
import json
import argparse
import numpy as np
import pypangolin as pango
from OpenGL.GL import *
from pytransform3d.rotations import axis_angle_from_matrix

from extractor.geotransforms import geodetic2enu
from extractor.mapping.geometry import from_twist


def load_data(workdir):
	# load dumped map and camera trajectory
	pose_graph = pickle.load(open(os.path.join(workdir, "mapping", "pose_graph.pkl"), "rb"))
	map_points = pickle.load(open(os.path.join(workdir, "mapping", "map_points.pkl"), "rb"))
	try:
		modules = pickle.load(open(os.path.join(workdir, "mapping", "modules_refined.pkl"), "rb"))
	except FileNotFoundError:
		modules = pickle.load(open(os.path.join(workdir, "mapping", "modules.pkl"), "rb"))
	module_corners = modules["corners"]
	module_centers = modules["centers"]

	# load GPS trajectory
	try:
		gps = json.load(open(os.path.join(workdir, "splitted", "gps", "gps.json"), "r"))
	except FileNotFoundError:
		gps_trajectory = []
	else:
		gps_trajectory = []
		gps_origin = pickle.load(open(os.path.join(workdir, "mapping", "reference_lla.pkl"), "rb"))
		for lon, lat, alt in gps:
			e, n, u = geodetic2enu(lat, lon, alt, *gps_origin)
			gps_trajectory.append([e, n, u])

	return pose_graph, map_points, module_corners, module_centers, gps_trajectory


def draw_camera_poses(poses, cam_scale, cam_aspect, color_line=(1.0, 0.6667, 0.0), color_frustrum=(1.0, 1.0, 1.0), draw_base=False):
	for R, t in [from_twist(pose) for pose in poses]:
		glPushMatrix()
		glTranslatef(*t)
		r = axis_angle_from_matrix(R) # returns x, y, z, angle
		r[-1] = r[-1]*180.0/np.pi  # rad -> deg
		glRotatef(r[3], r[0], r[1], r[2])  # angle, x, y, z

		if draw_base:
			glBegin(GL_LINES)
			glColor3f(1.0, 0, 0)
			glVertex3f(0, 0, 0)
			glVertex3f(cam_scale, 0, 0)

			glColor3f(0, 1.0, 0)
			glVertex3f(0, 0, 0)
			glVertex3f(0, cam_scale, 0)

			glColor3f(0, 0, 1.0)
			glVertex3f(0, 0, 0)
			glVertex3f(0, 0, cam_scale)
			glEnd()

		glLineWidth(2.0)
		glBegin(GL_LINE_LOOP)
		glColor3f(*color_frustrum)
		glVertex3f(-1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
		glVertex3f(1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
		glVertex3f(1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
		glVertex3f(-1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
		glEnd()

		glBegin(GL_LINES)
		glColor3f(*color_frustrum)
		glVertex3f(-1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
		glVertex3f(0, 0, 0)
		glVertex3f(1.0*cam_scale, -1.0*cam_scale/cam_aspect, 0.75*cam_scale)
		glVertex3f(0, 0, 0)
		glVertex3f(1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
		glVertex3f(0, 0, 0)
		glVertex3f(-1.0*cam_scale, 1.0*cam_scale/cam_aspect, 0.75*cam_scale)
		glVertex3f(0, 0, 0)
		glEnd()
		glLineWidth(1.0)

		glPopMatrix()

	# connect camera poses with a line
	glLineWidth(2.0)
	glBegin(GL_LINE_STRIP)
	glColor3f(*color_line)
	for _, t in [from_twist(pose) for pose in poses]:
		glVertex3f(t[0, 0], t[1, 0], t[2, 0])
	glEnd()
	glLineWidth(1.0)


def draw_map_points(map_points, color=(0.5, 0.5, 0.5), size=2, subsample=1):
	glPointSize(size)
	glColor3f(*color)
	glBegin(GL_POINTS)
	for i, (p_x, p_y, p_z) in enumerate(zip(map_points[:, 0], map_points[:, 1], map_points[:, 2])):
		glVertex3f(p_x, p_y, p_z)
	glEnd()


def draw_gps_track(gps_positions, size=5.0, color=(1.0, 0.0, 0.0)):
	glLineWidth(2.0)
	glBegin(GL_LINE_STRIP)
	glColor3f(*color)
	for t in gps_positions:
		glVertex3f(t[0], t[1], t[2])
	glEnd()
	glLineWidth(1.0)

	glPointSize(size)
	glBegin(GL_POINTS)
	glColor3f(*color)
	for t in gps_positions:
		glVertex3f(t[0], t[1], t[2])
	glEnd()


def draw_pv_modules(module_corners, module_centers, color_edges=(1.0, 0.0, 0.0), color_center=(0.0, 1.0, 0.0)):
	glPointSize(5)  # 10
	glBegin(GL_POINTS)
	glColor3f(*color_edges)
	for _, points in module_corners.items():
		for p_x, p_y, p_z in zip(points[:, 0], points[:, 1], points[:, 2]):
			glVertex3f(p_x, p_y, p_z)

	glColor3f(*color_center)
	for _, points in module_centers.items():
		for p_x, p_y, p_z in zip(points[:, 0], points[:, 1], points[:, 2]):
			glVertex3f(p_x, p_y, p_z)
	glEnd()

	glLineWidth(2.0)  # 1.0
	glColor3f(*color_edges)
	for _, points in module_corners.items():
		glBegin(GL_LINE_LOOP)
		for p_x, p_y, p_z in zip(points[:, 0], points[:, 1], points[:, 2]):
			glVertex3f(p_x, p_y, p_z)
		glEnd()
	glLineWidth(1.0)


def draw_origin():
	"""Draw origin coordinate system (red: x, green: y, blue: z)"""
	glBegin(GL_LINES)
	glColor3f(1.0, 0, 0)
	glVertex3f(0, 0, 0)
	glVertex3f(3, 0, 0)

	glColor3f(0, 1.0, 0)
	glVertex3f(0, 0, 0)
	glVertex3f(0, 3, 0)

	glColor3f(0, 0, 1.0)
	glVertex3f(0, 0, 0)
	glVertex3f(0, 0, 3)
	glEnd()


def draw_ground_plane(ground_plane_z):
	glDisable(GL_TEXTURE_2D)
	glBegin(GL_QUADS)
	glNormal3f(0.0, 1.0, 0.0)
	z0 = ground_plane_z
	repeat = 20
	for y in range(repeat):
		yStart = 100.0 - y*10.0
		for x in range(repeat):
			xStart = x*10.0 - 100.0
			if ((y % 2) ^ (x % 2)):
				glColor3f(0.16, 0.16, 0.16)
			else:
				glColor3f(0.78, 0.78, 0.78)
			glVertex3f(xStart, yStart, z0)
			glVertex3f(xStart + 10.0, yStart, z0)
			glVertex3f(xStart + 10.0, yStart - 10.0, z0)
			glVertex3f(xStart, yStart - 10.0, z0)
	glEnd()


def plot(args):

	pose_graph, map_points, module_corners, module_centers, gps_trajectory = load_data(args.workdir)

	win = pango.CreateWindowAndBind("pySimpleDisplay", 1600, 900)
	glEnable(GL_DEPTH_TEST)

	aspect = 1600/900
	cam_scale = 0.5
	cam_aspect = aspect

	pm = pango.ProjectionMatrix(
		1600,  # width
		900,  # height
		1000,  # fu
		1000,  # fv
		800,  # u0
		450,  # v0
		0.1,  # z near
		1000)  # z far
	#mv = pango.ModelViewLookAt(0, 0, 100, 0, 0, 0, pango.AxisY)
	#mv = pango.ModelViewLookAt(-60, 30, -1.5, -60, 30, -20, pango.AxisY)
	#mv = pango.ModelViewLookAt(-55, -35, 60, -5, 30, -40, pango.AxisZ)  # top area at an angle
	mv = pango.ModelViewLookAt(50, -65, 240, 50, -65, -20, pango.AxisY)  # top down whole plant
	s_cam = pango.OpenGlRenderState(pm, mv)

	handler = pango.Handler3D(s_cam)
	d_cam = pango.CreateDisplay().SetBounds(pango.Attach(0),
											pango.Attach(1),
											pango.Attach(0),
											pango.Attach(1),
											-aspect).SetHandler(handler)

	# find z position of ground plane
	ground_plane_z = np.quantile(map_points[:, -1], 0.05)
	poses = [pose_graph.nodes[n]["pose"] for n in pose_graph]

	while not pango.ShouldQuit():
		glClearColor(1.0, 1.0, 1.0, 1.0)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		d_cam.Activate(s_cam)
		glMatrixMode(GL_MODELVIEW)

		#draw_ground_plane(ground_plane_z)
		draw_origin()

		if args.show_map_points:
			draw_map_points(map_points, color=(0.5, 0.5, 0.5), size=2)

		if args.show_camera_poses:
			draw_camera_poses(poses, cam_scale, cam_aspect, color_line=(0.0, 0.5, 0.7), color_frustrum=(0.2, 0.2, 0.2))

		if args.show_gps_track:
			draw_gps_track(gps_trajectory)

		if args.show_modules:
			draw_pv_modules(module_corners, module_centers)

		pango.FinishFrame()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Visualize OpenSfM reconstruction of plant")
	parser.add_argument("workdir", type=str, help="workdir of the plant you want to visualize")

	parser.add_argument('--hide-map-points', dest='show_map_points', action='store_false')
	parser.add_argument('--show-map-points', dest='show_map_points', action='store_true')
	parser.set_defaults(show_map_points=True)

	parser.add_argument('--hide-modules', dest='show_modules', action='store_false')
	parser.add_argument('--show-modules', dest='show_modules', action='store_true')
	parser.set_defaults(show_modules=True)

	parser.add_argument('--hide-gps-track', dest='show_gps_track', action='store_false')
	parser.add_argument('--show-gps-track', dest='show_gps_track', action='store_true')
	parser.set_defaults(show_gps_track=False)

	parser.add_argument('--hide-camera-poses', dest='show_camera_poses', action='store_false')
	parser.add_argument('--show-camera-poses', dest='show_camera_poses', action='store_true')
	parser.set_defaults(show_camera_poses=True)

	args = parser.parse_args()

	plot(args)
