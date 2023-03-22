function [] = folder_creation (of)
if ~exist([of '\Images'], 'dir')
    mkdir([of '\Images'])
end
if ~exist([of '\Images_orig'], 'dir')
    mkdir([of '\Images_orig'])
end
if ~exist([of '\Vessels'], 'dir')
    mkdir([of '\Vessels'])
end
if ~exist([of '\Disc'], 'dir')
    mkdir([of '\Disc'])
end
if ~exist([of '\Cup'], 'dir')
    mkdir([of '\Cup'])
end
if ~exist([of '\Fov'], 'dir')
    mkdir([of '\Fov'])
end
if ~exist([of '\Images_crop'], 'dir')
    mkdir([of '\Images_crop'])
end
if ~exist([of '\Images_orig_crop'], 'dir')
    mkdir([of '\Images_orig_crop'])
end
if ~exist([of '\Disc_crop'], 'dir')
    mkdir([of '\Disc_crop'])
end
if ~exist([of '\Cup_crop'], 'dir')
    mkdir([of '\Cup_crop'])
end
end