function [] = save_landmark_fig(bs, posemap, im, landmark, imname )
     bshownumber = true;
     bdrawpose = true;
     bvisible = false;
     c = bs(1).c;    %we assume there is only one face in the image.
     val_pose = posemap(c);
     str_pose = sprintf('%d',val_pose);        
     hfig = U21b_DrawLandmarks_Points_ReturnHandle(im,landmark,str_pose,bshownumber,bdrawpose,bvisible);
     fn_save = sprintf(imname);
     saveas(hfig, fn_save);
     close(hfig);
end