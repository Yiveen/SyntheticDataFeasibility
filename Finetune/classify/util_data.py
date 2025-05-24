from os.path import join as ospj
import math
import json
import codecs
import numpy as np
from PIL import Image
from munch import Munch as mch
import torch
import torchvision as tv
import torchvision.transforms.functional as F
from typing import List, Optional, Tuple, Union
from torch import Tensor
import numbers
from collections.abc import Sequence
import random
from PIL import ImageFilter, ImageOps


SUBSET_NAMES = {
    'oxford_pets': [
        'Abyssinian', 'American Bulldog', 'American Pit Bull Terrier', 'Basset Hound', 'Beagle',
        'Bengal', 'Birman', 'Bombay', 'Boxer', 'British Shorthair',
        'Chihuahua', 'Egyptian Mau', 'English Cocker Spaniel', 'English Setter', 'German Shorthaired',
        'Great Pyrenees', 'Havanese', 'Japanese Chin', 'Keeshond', 'Leonberger',
        'Maine Coon', 'Miniature Pinscher', 'Newfoundland', 'Persian', 'Pomeranian',
        'Pug', 'Ragdoll', 'Russian Blue', 'Saint Bernard', 'Samoyed',
        'Scottish Terrier', 'Shiba Inu', 'Siamese', 'Sphynx', 'Staffordshire Bull Terrier',
        'Wheaten Terrier', 'Yorkshire Terrier'
    ],
    'waterbirds': [
        'landbird', 'waterbird'
    ],
    'waterbirds_nobias': [
        'landbird', 'waterbird'
    ],
    'fgvc_aircraft': [
        '707-320', '727-200', '737-200', '737-300', '737-400',
        '737-500', '737-600', '737-700', '737-800', '737-900',
        '747-100', '747-200', '747-300', '747-400', '757-200',
        '757-300', '767-200', '767-300', '767-400', '777-200',
        '777-300', 'A300B4', 'A310', 'A318', 'A319',
        'A320', 'A321', 'A330-200', 'A330-300', 'A340-200',
        'A340-300', 'A340-500', 'A340-600', 'A380', 'ATR-42',
        'ATR-72', 'An-12', 'BAE 146-200', 'BAE 146-300', 'BAE-125',
        'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47', 'CRJ-200',
        'CRJ-700', 'CRJ-900', 'Cessna 172', 'Cessna 208', 'Cessna 525',
        'Cessna 560', 'Challenger 600', 'DC-10', 'DC-3', 'DC-6',
        'DC-8', 'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6',
        'DHC-8-100', 'DHC-8-300', 'DR-400', 'Dornier 328', 'E-170',
        'E-190', 'E-195', 'EMB-120', 'ERJ 135', 'ERJ 145',
        'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16A/B', 'F/A-18', 'Falcon 2000',
        'Falcon 900', 'Fokker 100', 'Fokker 50', 'Fokker 70', 'Global Express',
        'Gulfstream IV', 'Gulfstream V', 'Hawk T1', 'Il-76', 'L-1011',
        'MD-11', 'MD-80', 'MD-87', 'MD-90', 'Metroliner',
        'Model B200', 'PA-28', 'SR-20', 'Saab 2000', 'Saab 340',
        'Spitfire', 'Tornado', 'Tu-134', 'Tu-154', 'Yak-42'
    ],
    'cars': [
        'AM General Hummer SUV 2000', 'Acura RL Sedan 2012', 'Acura TL Sedan 2012', 'Acura TL Type-S 2008', 'Acura TSX Sedan 2012',
        'Acura Integra Type R 2001', 'Acura ZDX Hatchback 2012', 'Aston Martin V8 Vantage Convertible 2012', 'Aston Martin V8 Vantage Coupe 2012', 'Aston Martin Virage Convertible 2012',
        'Aston Martin Virage Coupe 2012', 'Audi RS 4 Convertible 2008', 'Audi A5 Coupe 2012', 'Audi TTS Coupe 2012', 'Audi R8 Coupe 2012',
        'Audi V8 Sedan 1994', 'Audi 100 Sedan 1994', 'Audi 100 Wagon 1994', 'Audi TT Hatchback 2011', 'Audi S6 Sedan 2011',
        'Audi S5 Convertible 2012', 'Audi S5 Coupe 2012', 'Audi S4 Sedan 2012', 'Audi S4 Sedan 2007', 'Audi TT RS Coupe 2012',
        'BMW ActiveHybrid 5 Sedan 2012', 'BMW 1 Series Convertible 2012', 'BMW 1 Series Coupe 2012', 'BMW 3 Series Sedan 2012', 'BMW 3 Series Wagon 2012',
        'BMW 6 Series Convertible 2007', 'BMW X5 SUV 2007', 'BMW X6 SUV 2012', 'BMW M3 Coupe 2012', 'BMW M5 Sedan 2010',
        'BMW M6 Convertible 2010', 'BMW X3 SUV 2012', 'BMW Z4 Convertible 2012', 'Bentley Continental Supersports Conv. Convertible 2012', 'Bentley Arnage Sedan 2009',
        'Bentley Mulsanne Sedan 2011', 'Bentley Continental GT Coupe 2012', 'Bentley Continental GT Coupe 2007', 'Bentley Continental Flying Spur Sedan 2007', 'Bugatti Veyron 16.4 Convertible 2009',
        'Bugatti Veyron 16.4 Coupe 2009', 'Buick Regal GS 2012', 'Buick Rainier SUV 2007', 'Buick Verano Sedan 2012', 'Buick Enclave SUV 2012',
        'Cadillac CTS-V Sedan 2012', 'Cadillac SRX SUV 2012', 'Cadillac Escalade EXT Crew Cab 2007', 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 'Chevrolet Corvette Convertible 2012',
        'Chevrolet Corvette ZR1 2012', 'Chevrolet Corvette Ron Fellows Edition Z06 2007', 'Chevrolet Traverse SUV 2012', 'Chevrolet Camaro Convertible 2012', 'Chevrolet HHR SS 2010',
        'Chevrolet Impala Sedan 2007', 'Chevrolet Tahoe Hybrid SUV 2012', 'Chevrolet Sonic Sedan 2012', 'Chevrolet Express Cargo Van 2007', 'Chevrolet Avalanche Crew Cab 2012',
        'Chevrolet Cobalt SS 2010', 'Chevrolet Malibu Hybrid Sedan 2010', 'Chevrolet TrailBlazer SS 2009', 'Chevrolet Silverado 2500HD Regular Cab 2012', 'Chevrolet Silverado 1500 Classic Extended Cab 2007',
        'Chevrolet Express Van 2007', 'Chevrolet Monte Carlo Coupe 2007', 'Chevrolet Malibu Sedan 2007', 'Chevrolet Silverado 1500 Extended Cab 2012', 'Chevrolet Silverado 1500 Regular Cab 2012',
        'Chrysler Aspen SUV 2009', 'Chrysler Sebring Convertible 2010', 'Chrysler Town and Country Minivan 2012', 'Chrysler 300 SRT-8 2010', 'Chrysler Crossfire Convertible 2008',
        'Chrysler PT Cruiser Convertible 2008', 'Daewoo Nubira Wagon 2002', 'Dodge Caliber Wagon 2012', 'Dodge Caliber Wagon 2007', 'Dodge Caravan Minivan 1997',
        'Dodge Ram Pickup 3500 Crew Cab 2010', 'Dodge Ram Pickup 3500 Quad Cab 2009', 'Dodge Sprinter Cargo Van 2009', 'Dodge Journey SUV 2012', 'Dodge Dakota Crew Cab 2010',
        'Dodge Dakota Club Cab 2007', 'Dodge Magnum Wagon 2008', 'Dodge Challenger SRT8 2011', 'Dodge Durango SUV 2012', 'Dodge Durango SUV 2007',
        'Dodge Charger Sedan 2012', 'Dodge Charger SRT-8 2009', 'Eagle Talon Hatchback 1998', 'FIAT 500 Abarth 2012', 'FIAT 500 Convertible 2012',
        'Ferrari FF Coupe 2012', 'Ferrari California Convertible 2012', 'Ferrari 458 Italia Convertible 2012', 'Ferrari 458 Italia Coupe 2012', 'Fisker Karma Sedan 2012',
        'Ford F-450 Super Duty Crew Cab 2012', 'Ford Mustang Convertible 2007', 'Ford Freestar Minivan 2007', 'Ford Expedition EL SUV 2009', 'Ford Edge SUV 2012',
        'Ford Ranger SuperCab 2011', 'Ford GT Coupe 2006', 'Ford F-150 Regular Cab 2012', 'Ford F-150 Regular Cab 2007', 'Ford Focus Sedan 2007',
        'Ford E-Series Wagon Van 2012', 'Ford Fiesta Sedan 2012', 'GMC Terrain SUV 2012', 'GMC Savana Van 2012', 'GMC Yukon Hybrid SUV 2012',
        'GMC Acadia SUV 2012', 'GMC Canyon Extended Cab 2012', 'Geo Metro Convertible 1993', 'HUMMER H3T Crew Cab 2010', 'HUMMER H2 SUT Crew Cab 2009',
        'Honda Odyssey Minivan 2012', 'Honda Odyssey Minivan 2007', 'Honda Accord Coupe 2012', 'Honda Accord Sedan 2012', 'Hyundai Veloster Hatchback 2012',
        'Hyundai Santa Fe SUV 2012', 'Hyundai Tucson SUV 2012', 'Hyundai Veracruz SUV 2012', 'Hyundai Sonata Hybrid Sedan 2012', 'Hyundai Elantra Sedan 2007',
        'Hyundai Accent Sedan 2012', 'Hyundai Genesis Sedan 2012', 'Hyundai Sonata Sedan 2012', 'Hyundai Elantra Touring Hatchback 2012', 'Hyundai Azera Sedan 2012',
        'Infiniti G Coupe IPL 2012', 'Infiniti QX56 SUV 2011', 'Isuzu Ascender SUV 2008', 'Jaguar XK XKR 2012', 'Jeep Patriot SUV 2012',
        'Jeep Wrangler SUV 2012', 'Jeep Liberty SUV 2012', 'Jeep Grand Cherokee SUV 2012', 'Jeep Compass SUV 2012', 'Lamborghini Reventon Coupe 2008',
        'Lamborghini Aventador Coupe 2012', 'Lamborghini Gallardo LP 570-4 Superleggera 2012', 'Lamborghini Diablo Coupe 2001', 'Land Rover Range Rover SUV 2012', 'Land Rover LR2 SUV 2012',
        'Lincoln Town Car Sedan 2011', 'MINI Cooper Roadster Convertible 2012', 'Maybach Landaulet Convertible 2012', 'Mazda Tribute SUV 2011', 'McLaren MP4-12C Coupe 2012',
        'Mercedes-Benz 300-Class Convertible 1993', 'Mercedes-Benz C-Class Sedan 2012', 'Mercedes-Benz SL-Class Coupe 2009', 'Mercedes-Benz E-Class Sedan 2012', 'Mercedes-Benz S-Class Sedan 2012',
        'Mercedes-Benz Sprinter Van 2012', 'Mitsubishi Lancer Sedan 2012', 'Nissan Leaf Hatchback 2012', 'Nissan NV Passenger Van 2012', 'Nissan Juke Hatchback 2012',
        'Nissan 240SX Coupe 1998', 'Plymouth Neon Coupe 1999', 'Porsche Panamera Sedan 2012', 'Ram C/V Cargo Van Minivan 2012', 'Rolls-Royce Phantom Drophead Coupe Convertible 2012',
        'Rolls-Royce Ghost Sedan 2012', 'Rolls-Royce Phantom Sedan 2012', 'Scion xD Hatchback 2012', 'Spyker C8 Convertible 2009', 'Spyker C8 Coupe 2009',
        'Suzuki Aerio Sedan 2007', 'Suzuki Kizashi Sedan 2012', 'Suzuki SX4 Hatchback 2012', 'Suzuki SX4 Sedan 2012', 'Tesla Model S Sedan 2012',
        'Toyota Sequoia SUV 2012', 'Toyota Camry Sedan 2012', 'Toyota Corolla Sedan 2012', 'Toyota 4Runner SUV 2012', 'Volkswagen Golf Hatchback 2012',
        'Volkswagen Golf Hatchback 1991', 'Volkswagen Beetle Hatchback 2012', 'Volvo C30 Hatchback 2012', 'Volvo 240 Sedan 1993', 'Volvo XC90 SUV 2007',
        'smart fortwo Convertible 2012'],
}


TEMPLATES_SMALL = [
    "a {}photo of a {}",
    "a {}rendering of a {}",
    "a {}cropped photo of the {}",
    "the {}photo of a {}",
    "a {}photo of a clean {}",
    "a {}photo of a dirty {}",
    "a dark {}photo of the {}",
    "a {}photo of my {}",
    "a {}photo of the cool {}",
    "a close-up {}photo of a {}",
    "a bright {}photo of the {}",
    "a cropped {}photo of a {}",
    "a {}photo of the {}",
    "a good {}photo of the {}",
    "a {}photo of one {}",
    "a close-up {}photo of the {}",
    "a {}rendition of the {}",
    "a {}photo of the clean {}",
    "a {}rendition of a {}",
    "a {}photo of a nice {}",
    "a good {}photo of a {}",
    "a {}photo of the nice {}",
    "a {}photo of the small {}",
    "a {}photo of the weird {}",
    "a {}photo of the large {}",
    "a {}photo of a cool {}",
    "a {}photo of a small {}",
]


class UnNormalize(object):
    def __init__(self, 
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        unnormed_tensor = torch.zeros_like(tensor)
        for i, (t, m, s) in enumerate(zip(tensor, self.mean, self.std)):
            unnormed_tensor[i] = t.mul(s).add(m)
            # The normalize code -> t.sub_(m).div_(s)
        return unnormed_tensor

unnorm = UnNormalize()


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = ospj(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = ospj(metadata_root, 'image_ids_proxy.txt')
    metadata.class_labels = ospj(metadata_root, 'class_labels.txt')
    return metadata


def get_image_ids(metadata, proxy=False):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    image_ids = []
    suffix = '_proxy' if proxy else ''
    with open(metadata['image_ids' + suffix]) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_class_labels(metadata):
    """
    class_labels.txt has the structure

    <path>,<integer_class_label>
    path/to/image1.jpg,0
    path/to/image2.jpg,1
    path/to/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels





class GaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __repr__(self):
        return "{}(p={}, radius_min={}, radius_max={})".format(
            self.__class__.__name__, self.p, self.radius_min, self.radius_max
        )

    def __call__(self, img):
        if random.random() <= self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __repr__(self):
        return "{}(p={})".format(self.__class__.__name__, self.p)

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img




