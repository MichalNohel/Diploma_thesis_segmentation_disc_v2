close all;clear all;clc;
% rmdir ('Cup','s');rmdir ('Disc','s');rmdir ('Fov','s');rmdir ('Images','s');rmdir ('Vessels','s')
path = pwd;
path= [path '\'];
% poz = strfind(path,'\');
% path = path(1:poz(end));
%%  SET
resolution = 35;  % kolik pixelu na stupen - kolem 30
output_folder = cd;

%%
folder_creation(output_folder);
%
% load_aria(resolution, path, output_folder);

% load_avrdb(resolution, path, output_folder);

% load_chasedb1(resolution, path, output_folder);

% load_diaretdb0(resolution, path, output_folder);

% load_diaretdb1(resolution, path, output_folder);

% load_drhagis(resolution, path, output_folder);

load_drishtigs(resolution, path, output_folder);

% load_drive(resolution, path, output_folder);

% load_eophtha(resolution, path, output_folder);

% load_eyepacs(resolution, path, output_folder);

% load_erlangen(resolution, path, output_folder);

% load_fire(resolution, path, output_folder);

% load_g1020(resolution, path, output_folder);

% load_heimed(resolution, path, output_folder);

% load_hrf(resolution, path, output_folder);

% load_idrid(resolution, path, output_folder);

% load_inspireavr(resolution, path, output_folder);

% load_iostar(resolution, path, output_folder);

% load_messidor(resolution, path, output_folder);

% load_onhsd(resolution, path, output_folder);

% load_rcrgbma(resolution, path, output_folder);

% load_refuge(resolution, path, output_folder);

% load_review(resolution, path, output_folder);

% load_ridb(resolution, path, output_folder);

% load_riga(resolution, path, output_folder);

% load_rimone(resolution, path, output_folder);

% load_roc(resolution, path, output_folder);

% load_stare(resolution, path, output_folder);

% load_uoadr(resolution, path, output_folder);

% load_vicavr(resolution, path, output_folder);


%%
load chirp
sound(y/10,Fs)