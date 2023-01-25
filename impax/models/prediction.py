class Prediction(object):
    """A prediction made by a model."""

    def __init__(self, model_config, observation, structured_implicit, embedding):
        self._observation = observation
        self._structured_implicit = structured_implicit
        self._embedding = embedding
        self._in_out_image = None
        self._model_config = model_config

    def has_embedding(self):
        return self._embedding is not None

    def export_signature_def(self):
        input_map = {}
        input_map["observation"] = self.observation.tensor
        output_map = {}
        if self.has_embedding:
            output_map["embedding"] = self.embedding
        output_map["structured_implicit_vector"] = self.structured_implicit.vector
        return {"inputs": input_map, "outputs": output_map}

    @property
    def embedding(self):
        return self._embedding

    @property
    def structured_implicit(self):
        return self._structured_implicit

    @property
    def observation(self):
        return self._observation
