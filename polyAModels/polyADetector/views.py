from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import numpy as np
import os
from django.conf import settings
# Create your views here.
def index(request):
    
    return render(request, 'index.html')



def process_form(request):
    if request.method == 'POST':
        chr_value = request.POST.get('chr')
        start_value = request.POST.get('start')
        end_value = request.POST.get('end')
        minus_strand = request.POST.get('minusStrand')
        sequence_value = request.POST.get('sequence')
        species_value = request.POST.get('species')
        if not sequence_value and not all([chr_value, start_value, end_value]):
            return HttpResponse("Invalid input: Either Sequence should be non-empty or Chr, Start, and End together should be non-empty.")
        fields = {
            "sequence": sequence_value,
            'chrom': chr_value,
            'position': start_value,
            'strand': minus_strand
        }

        print("Chr:", chr_value)
        print("Start:", start_value)
        print("End:", end_value)
        print("Minus Strand:", minus_strand)
        print("Sequence:", sequence_value)
        print("Species:", species_value)

        predictions = predict(fields)
        print('predictions', predictions)
        context = {
            'predictions': predictions
        }
        return JsonResponse(context)

def predict(fields):

    from . import PolyaID_PolyaStrength_utilities as utils
    PolyaID_model = os.path.join(settings.BASE_DIR, 'polyADetector/PolyaID_model.h5')
    PolyaStrength_model = os.path.join(settings.BASE_DIR, 'polyADetector/PolyaStrength_model.h5')
    genome_path = os.path.join(settings.BASE_DIR, 'genome.fa')
    chromesizes_path = os.path.join(settings.BASE_DIR, 'chrom.sizes')
    genome = utils.load_genome(genome_path)
    chrom_sizes = utils.get_chrom_size(chromesizes_path)

    # Load the TensorFlow model
    polyaID       = utils.make_polyaid_model(PolyaID_model)
    polyaStrength = utils.make_polyastrength_model(PolyaStrength_model)

    # Process input data (e.g., extract features from request)
    sequence = fields["sequence"]

    if sequence == '' or sequence is None:
        position = int(fields['position'])
        chrom = fields['chrom']
        chrom = 'chr' + chrom
        if fields['strand'] is None:
            strand = '+'
        else:
            strand = '-'
        sequence = utils.get_window(genome, chrom_sizes, chrom, position, strand)
    
    print('seq in predict ', sequence)
    
        
        

    # Encode input sequence to matrix
    encoding = utils.generate_data(sequence)['predict'][0]

    # Generate predictions
    polyaID_prediction     = polyaID.predict(encoding)
    polyaID_classification = polyaID_prediction[0][0][0]
    polyaID_rawcleavage    = polyaID_prediction[1].flatten()

    polyaID_subtracted = polyaID_rawcleavage - 0.02
    polyaID_subtracted[polyaID_subtracted <= 0] = 0
    polyaID_normcleavage = polyaID_subtracted / np.sum(polyaID_subtracted) if (np.sum(polyaID_subtracted) > 0) else np.asarray([0]*50)

    polyaStrength_score = polyaStrength.predict(encoding)[0][0]

    # Return 3 predictions for each sequence - classification, cleavage probability vector, and strength
    predictions = {
        'classification': float(polyaID_classification), 
        'clevage_prob_vector': polyaID_normcleavage.round(3).tolist(), 
        
        'strength': float(polyaStrength_score),
        'sequence': sequence}

    return predictions


def browser(request):
    return render(request, 'browser.html')