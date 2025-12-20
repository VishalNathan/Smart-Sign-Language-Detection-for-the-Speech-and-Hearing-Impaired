function image_file_prediction(aslNet, inputSize)
    % IMAGE_FILE_PREDICTION
    % This function allows the user to select an image file,
    % preprocesses the image, performs classification using
    % a trained CNN (aslNet), and displays the prediction result.

    %--------------------------------------------------------------
    % 1. Select an image file from the system
    %--------------------------------------------------------------
    [file, path] = uigetfile({'*.jpg;*.png;*.jpeg'}, 'Select an Image File');
    
    % If user cancels file selection, exit function
    if isequal(file, 0)
        disp('No image selected.');
        return;
    end

    try
        %----------------------------------------------------------
        % 2. Read the selected image
        %----------------------------------------------------------
        img = imread(fullfile(path, file));

        %----------------------------------------------------------
        % 3. Handle grayscale images (convert to RGB)
        %----------------------------------------------------------
        if size(img, 3) == 1
            img = cat(3, img, img, img);
        end

        %----------------------------------------------------------
        % 4. Handle RGBA images (remove alpha channel)
        %----------------------------------------------------------
        if size(img, 3) == 4
            img = img(:, :, 1:3);
        end

        %----------------------------------------------------------
        % 5. Resize image to match network input size
        %----------------------------------------------------------
        img = imresize(img, inputSize(1:2));

        %----------------------------------------------------------
        % 6. Convert image to single precision if required
        %----------------------------------------------------------
        if ~isa(img, 'single') && ~isa(img, 'double')
            img = im2single(img);
        end

        %----------------------------------------------------------
        % 7. Classify image and obtain confidence scores
        %----------------------------------------------------------
        [prediction, scores] = classify(aslNet, img);

        % Extract highest confidence score
        confidence = max(scores) * 100;

        %----------------------------------------------------------
        % 8. Display the image and prediction result
        %----------------------------------------------------------
        figure('Name', 'Image Prediction', 'NumberTitle', 'off');
        imshow(img);
        title(sprintf('Predicted Sign: %s (%.2f%%)', ...
            char(prediction), confidence), 'FontSize', 18);

        %----------------------------------------------------------
        % 9. Print prediction result in Command Window
        %----------------------------------------------------------
        fprintf('✅ Predicted Sign: %s\n', char(prediction));
        fprintf('📊 Confidence: %.2f%%\n', confidence);

    catch ME
        %----------------------------------------------------------
        % 10. Error handling (prevents function crash)
        %----------------------------------------------------------
        warning('Prediction failed: %s', ME.message);
    end
end
