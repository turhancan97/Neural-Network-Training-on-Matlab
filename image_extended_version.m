clc
clear all
%% Image Recognition Extended
image1 = imread("image1.jpg"); % Reading image - 1
image2 = imread("image2.jpg"); % Reading image - 2
image3 = imread("image3.jpg"); % Reading image - 3
image4 = imread("image4.jpg"); % Reading image - 4
image5 = imread("image5.jpg"); % Reading image - 5
%% Size Check
size(image1)
size(image2)
size(image3)
size(image4)
size(image5)
%%  From rgb to gray scale - We want 2 dimensions
image1_gray = rgb2gray(image1); 
imshow(image1_gray)
image2_gray = rgb2gray(image2); 
imshow(image2_gray)
image3_gray = rgb2gray(image3); 
imshow(image3_gray)
image4_gray = rgb2gray(image4); 
imshow(image4_gray)
image5_gray = rgb2gray(image5); 
imshow(image5_gray)
%% Let's resize for the ANN 
P1=imresize(image1_gray,[20 20]);
P1=P1(:);
P2=imresize(image2_gray,[20 20]);
P2=P2(:);
P3=imresize(image3_gray,[20 20]);
P3=P3(:);
P4=imresize(image4_gray,[20 20]);
P4=P4(:);
P5=imresize(image5_gray,[20 20]);
P5=P5(:);
% PA' -> Tanspose
P = [P1 P2 P3 P4 P5];
%% Target Vector
T1 = [1 0 0 0 0];
T2 = [0 1 0 0 0];
T3 = [0 0 1 0 0];
T4 = [0 0 0 1 0];
T5 = [0 0 0 0 1];
T = [T1' T2' T3' T4' T5'];
%% NN
range = [zeros(400,1) 255*ones(400,1)];
net = newff(range,[20 5],{'tansig' ,'tansig'});
net.trainParam.epochs = 10;
[net, tr, Y, E] = train(net,P,T);
% net = train(net, P, T, 'useGPU', 'yes'); % for faster train
%% Check Model
ans1 = sim(net,P1);
ans2 = sim(net,P2);
ans3 = sim(net,P3);
ans4 = sim(net,P4);
ans5 = sim(net,P5);
ansALL = sim(net,P);
% YPred = classify(net,imdsValidation);