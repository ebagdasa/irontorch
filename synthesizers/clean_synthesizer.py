from synthesizers.pattern_synthesizer import PatternSynthesizer


class CleanSynthesizer(PatternSynthesizer):
    name = 'Clean'

    def get_label(self, input_tensor, target_tensor):
        target_label = self.params.backdoor_labels[self.name]
        return target_label
