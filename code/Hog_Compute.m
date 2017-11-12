function feature = Hog_Compute(im,hog_cell_size)
if size(im,3)==3
    im=rgb2gray(im);
end
im=double(im);
rows=size(im,1);
cols=size(im,2);
Ix=im; 
Iy=im; 
for i=1:rows-2
    Iy(i,:)=(im(i,:)-im(i+2,:));
end
for i=1:cols-2
    Ix(:,i)=(im(:,i)-im(:,i+2));
end
angle=atand(Ix./Iy); 
angle=imadd(angle,90); 
magnitude=sqrt(Ix.^2 + Iy.^2);
feature=[]; 
for i = 0: rows/hog_cell_size - 2
    for j= 0: cols/hog_cell_size -2        
        mag_patch = magnitude(hog_cell_size*i+1 : hog_cell_size*i+hog_cell_size*2 , hog_cell_size*j+1 : hog_cell_size*j+hog_cell_size*2); 
        ang_patch = angle(hog_cell_size*i+1 : hog_cell_size*i+hog_cell_size*2 , hog_cell_size*j+1 : hog_cell_size*j+hog_cell_size*2);        
        patch_features=[];        
        for x= 0:1
            for y= 0:1
                angle_patch =ang_patch(hog_cell_size*x+1:hog_cell_size*x+hog_cell_size, hog_cell_size*y+1:hog_cell_size*y+hog_cell_size);
                magnitude_patch   =mag_patch(hog_cell_size*x+1:hog_cell_size*x+hog_cell_size, hog_cell_size*y+1:hog_cell_size*y+hog_cell_size);
                histogram  =zeros(1,6);               
                for columns=1:hog_cell_size
                    for rows=1:hog_cell_size                      
                        between_angle= angle_patch(columns,rows);                        
                        if between_angle>10 && between_angle<=40
                            histogram(1)=histogram(1)+ magnitude_patch(columns,rows)*(40-between_angle)/30;
                            histogram(2)=histogram(2)+ magnitude_patch(columns,rows)*(between_angle-10)/30;
                        elseif between_angle>40 && between_angle<=70
                            histogram(2)=histogram(2)+ magnitude_patch(columns,rows)*(70-between_angle)/30;                 
                            histogram(3)=histogram(3)+ magnitude_patch(columns,rows)*(between_angle-40)/30;
                        elseif between_angle>70 && between_angle<=100
                            histogram(3)=histogram(3)+ magnitude_patch(columns,rows)*(100-between_angle)/30;
                            histogram(4)=histogram(4)+ magnitude_patch(columns,rows)*(between_angle-70)/30;
                        elseif between_angle>100 && between_angle<=130
                            histogram(4)=histogram(4)+ magnitude_patch(columns,rows)*(130-between_angle)/30;
                            histogram(5)=histogram(5)+ magnitude_patch(columns,rows)*(between_angle-100)/30;
                        elseif between_angle>130 && between_angle<=160
                            histogram(5)=histogram(5)+ magnitude_patch(columns,rows)*(160-between_angle)/30;
                            histogram(6)=histogram(6)+ magnitude_patch(columns,rows)*(between_angle-130)/30;                        
                        elseif between_angle>=0 && between_angle<=10
                            histogram(1)=histogram(1)+ magnitude_patch(columns,rows)*(between_angle+10)/30;
                            histogram(6)=histogram(6)+ magnitude_patch(columns,rows)*(10-between_angle)/30;
                        elseif between_angle>160 && between_angle<=180
                            histogram(6)=histogram(6)+ magnitude_patch(columns,rows)*(180-between_angle)/30;
                            histogram(1)=histogram(1)+ magnitude_patch(columns,rows)*(between_angle-160)/30;
                        end                       
                
                    end
                end
                patch_features=[patch_features histogram];                                 
            end
        end 
        norm_value = sqrt(norm(patch_features)^2+.01);
        patch_features=patch_features/norm_value;               
        feature=[feature patch_features]; 
    end
end
end

    