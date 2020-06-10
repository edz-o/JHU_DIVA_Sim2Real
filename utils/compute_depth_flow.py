import matplotlib.pyplot as plt
import os
import cv2
import pdb
import numpy as np


def compute_flow():
  while(1):
      ret, frame2 = cap.read()
      next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

      flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

      mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
      hsv[...,0] = ang*180/np.pi/2
      hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
      rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

      cv2.imshow('frame2',rgb)
      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break
      elif k == ord('s'):
          cv2.imwrite('opticalfb.png',frame2)
          cv2.imwrite('opticalhsv.png',rgb)
      prvs = next

  cv2.destroyAllWindows()

def main():
  in_ = '/data/ue_data/sim_meva_20200604_6classes_depth'
  out_ = '/data/ue_data/sim_meva_20200604_6classes_depth_flow_bugfix'
  count = 0
  for inst in os.listdir(in_):
    frames = sorted(os.listdir(os.path.join(in_,inst)),key=lambda x:int(x[6:-4]))
    prvs = cv2.imread(os.path.join(in_, inst, frames[0]))[:, :, 0]
    plt.ion()
    ind = 0
    for frame in frames[1:]:
      next_ = cv2.imread(os.path.join(in_, inst, frame))[:, :, 0]
      flow = cv2.calcOpticalFlowFarneback(prvs,next_, None, 0.5, 3, 15, 3, 5, 1.2, 0)
      if not os.path.exists(os.path.join(out_,inst)):
        os.makedirs(os.path.join(out_,inst))
      np.save(os.path.join(out_,inst,'depthflow_{:05d}'.format(ind)),flow)
      prvs = next_
      ind += 1
    np.save(os.path.join(out_, inst, 'depthflow_{:05d}'.format(ind)), flow)
    count += 1
    print('{:d}/{:d}'.format(count,len(os.listdir(in_))))

    #   vis = np.zeros((flow.shape[0],flow.shape[1],3))
    #   flow = flow-flow.min()
    #   flow = flow / flow.max()
    #   flow *= 255
    #   vis[:,:,0] = flow[:,:,0]
    #   vis[:, :, 1] = flow[:, :, 1]
    #   vis[:, :, 2] = 127
    #   ax = plt.subplot(1,3,1)
    #   ax.imshow(prvs)
    #   ax = plt.subplot(1, 3, 2)
    #   ax.imshow(next_)
    #   ax = plt.subplot(1, 3, 3)
    #   ax.imshow(vis.astype(np.uint8))
    #   plt.show()
    #   plt.pause(0.1)
    # pdb.set_trace()

if __name__ == '__main__':

  main()
