README - You annotation layer for AMI corpus in NXT format

The directory youUsages contains the annotations of a number of AMI
meetings in NXT format.  The coding is based on the you annotations
from Gupta et al. 2007 (CALO project) and were used in a number of
subsequent research papers by various authors (see Purver et al.,
SigDial 2009).

They were provided by:
Matthew Purver 
Interaction, Media and Communication
Department of Computer Science
Queen Mary University of London, UK 

The code of you-types as used in the various publications
on ``you'' usages  
(see Gupta et al 2005 for an explanation if these aren't clear).  

0 = generic
1 = referential singular
2 = referential plural
4 = reported referential
5 = discourse marker ("you know")

(see also the NOTE at the end of this file)

Here is the code used in the NXT ontology for you-types.
Note that there are two subtypes of PLURAL DEICTIC usage.

"you_0" name="you-root" 
    "you_1" name="DEICTIC" abbrev="DEIC" gloss="deictic referential"
        "you_11" name="SINGULAR" abbrev="SING" gloss="singular"
        "you_12" name="PLURAL" abbrev="PLU" gloss="plural"
            "you_121" name="COLLECTIVE" abbrev="COLL" gloss="collective"
            "you_122" name="DISTRIBUTIVE" abbrev="DIST" gloss="distributive"
    "you_2" name="GENERIC" abbrev="GEN" gloss="general"
    "you_3" name="DISCOURSE" abbrev="DIS" gloss="discourse marker"


To use ``you'' annotation layer of the AMI corpus do the following:

Add the ontology file  you-types.xml to the ontologies directory


Add the following snippet to your nxt AMI metadata file

	<ontology attribute-name="name"
            description="'You' types for tagging the english pronoun 'you'"
            element-name="you-type" filename="you-types" name="you-types">
            <!-- Gloss: a short textual description of the you type -->
            <attribute name="gloss" value-type="string"/>
            <attribute name="abbrev" value-type="string"/>
        </ontology>
        

Add the following snippet to your resources.xml file
 
  <resource-type coding="you-types">
   <resource id="youtypes" description="you-usage types used for all manual annotation" type="manual" path="ontologies" />
  </resource-type>
  


Acknowledgement
thanks to the people from CSLI, Stanford and to Matthew Purver (QMUL),
for their efforts and permission.

Rieks op den Akker
University of Twente
Enschede
the Netherlands
--------------------

NOTE by Rieks:
In the original you annotations in a txt form a * is used when the
annotator wasn't sure.

This concerns the "difficult" cases, the annotation does tell you a
little bit more than that: the position of the * character tells you
in which "direction" the annotator was leaning. So 1* means they went
for 1, but considered 2; whereas *1 means they went for 1, but
considered 0.  The *-options was not translated in the NXT
annotations: we just removed the *.
