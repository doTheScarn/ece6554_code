% VL_DEMO_SIFT_MATCH  Demo: SIFT: basic matching

%pfx = fullfile(vl_root,'figures','demo') ;
%randn('state',0) ;
%rand('state',0) ;
figure(1) ; clf ;

% --------------------------------------------------------------------
%                                                    Create image pair
% --------------------------------------------------------------------

Im_name = ['sift-test-gary.jpg'];
Im = imread(fullfile(vl_root,'data', Im_name)) ;

templates = {'template-bottle.jpg','template-candle.jpg','template-chimney.jpg',...
    'template-cooler.jpg','template-fork.jpg','template-knife.jpg','template-newspaper.jpg',...
    'template-plate.jpg','template-ribeye.jpg','template-steak.jpg'} ;
templates = {'template-chimney.jpg'}; % Just doing one template for now

for i = 1:length(templates)
    Im_template{i} = imread(fullfile(vl_root,'data',templates{i})) ;
end

% --------------------------------------------------------------------
%                                           Extract features and match
% --------------------------------------------------------------------

[fa,da] = vl_sift(im2single(rgb2gray(Im))) ;

for i = 1:length(templates)
    [fb{i},db{i}] = vl_sift(im2single(rgb2gray(Im_template{i}))) ;
    [matches{i}, scores{i}] = vl_ubcmatch(da,db{i}) ;
    [drop, perm] = sort(scores{i}, 'descend') ;
    matches{i} = matches{i}(:, perm) ;
    scores{i}  = scores{i}(perm) ;
end

% --------------------------------------------------------------------
%                                                           Make Plots
% --------------------------------------------------------------------
figure(1) ; clf ;
imagesc(Im) ;
axis image off ;
%vl_demo_print('sift_match_1', 1) ;

figure(2) ; clf ;
imagesc(Im) ; 

i=1;
xa = fa(1,matches{i}(1,:)) ;
%xb = fb(1,matches{i}(2,:)) + size(Ia,2) ;
ya = fa(2,matches{i}(1,:)) ;
%yb = fb(2,matches{i}(2,:)) ;

hold on ;
%h = line([xa ; xb], [ya ; yb]) ;
%set(h,'linewidth', 1, 'color', 'b') ;

h = vl_plotframe(fa(:,matches{i}(1,:))) ;
%fb(1,:) = fb(1,:) + size(Ia,2) ;
%vl_plotframe(fb(:,matches(2,:))) ;
axis image off ;

%vl_demo_print('sift_match_2', 1) ;

% Gary: Added this from vl_demo_kdtree_sift.m
maxNumComparisonsRange = [1 10 50 100 200 300 400] ;
maxNumComparisonsRange = [400] ; % Just one value for demo
numTreesRange = [1 2 5 10] ;
numTreesRange = [10]; % Just one value for demo
D = vl_alldist(single(da),single(db{i})) ; % USES A WHOLE LOT OF MEMORY IF TOO MANY KEYS
[drop, best] = min(D, [], 1) ;
for ti=1:length(numTreesRange)
    for vi=1:length(maxNumComparisonsRange)
        v = maxNumComparisonsRange(vi) ;
        t = numTreesRange(ti) ;
        % Note [d1,d2] replaced with [da,db{i}]. Order matters here, or
        % there are errors on setting errors_mean below. I think it's
        % because it may be expecting the first array (da) to have more
        % points (bigger image w more keys) than the second (db)
        kdtree = vl_kdtreebuild(single(da), ...
            'verbose', ...
            'thresholdmethod', 'mean', ...  %note consider trying median
            'numtrees', t) ;
        [ii, d] = vl_kdtreequery(kdtree, single(da), single(db{i}), ... 
            'verbose', ...
            'maxcomparisons', v) ;
            % Returns INDEX, DIST of da nearest db{i}
            % Consider looking at the 'numneighbors', NN argument
        errors_mean(vi,ti) = sum(double(ii) ~= best) / length(best) ;
        errorsD_mean(vi,ti) = mean(abs(d - drop) ./ drop) ;
    end
end

h2 = vl_plotframe(fa(:,ii<2000)); % this is tiny scattered dots b/c it finds the "best" matches but no clustering.
  % I think this is "right" but the problem is there's no clustering yet.
set(h2,'color','red')

keyboard
[C, A] = vl_ikmeans(da(:,ii),5);


% Plot the first template with SIFT Keys
figure(3)
imagesc(Im_template{1})
ht = vl_plotframe(fb{i}(:,matches{i}(2,:)));

  % This is on the right track, but it just bins the raw keys, not their distance metric. 
% How do I take the kmeans or ikmeans or hikmeans output and map to features?

% So what am i trying to cluster? Not the raw SIFT keys; not all the best matches (aka nearest neighbor, aka smallest error).
% I THINK matches gives me the corresponding indices of fa/fb. 
% SO MAYBE take the x,y positions of matches, do kmeans, then rank the Score for each cluster.

% REMOVE THE double for loop and replace with static values
% I should be using the pre-matched matches and scores to do this, much smaller lists.