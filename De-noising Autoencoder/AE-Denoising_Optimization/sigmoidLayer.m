classdef sigmoidLayer < nnet.layer.Layer
    
    methods
        
        function layer = sigmoidLayer(options)
            arguments
            options.Name (1,1) string = ""
        end
        layer.Name = options.Name;
    end
    
    function Z = predict(~, X)
    Z = 1./(1 + exp(-X));
    end
    
    function dLdX = backward(~, ~, Z, dLdZ, ~)
    dLdX = dLdZ .* (Z .* (1 - Z));
    end
    
end

end