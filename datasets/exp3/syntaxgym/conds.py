# FORMAT: each test suite has an associated list of condition orderings.
# Each condition ordering is a tuple of conditions, where the 
# sentence in the first condition should have higher probability than 
# the sentence in the second condition.

CONDITION_ORDERINGS = {
    "center_embed": [
        ("plaus", "implaus")
    ],
    "center_embed_mod": [
        ("plaus", "implaus")
    ],
    "cleft": [
        ("np_match", "np_mismatch"),
        ("vp_match", "vp_mismatch")
    ],
    "cleft_modifier": [
        ("np_match", "np_mismatch"),
        ("vp_match", "vp_mismatch")
    ],
    "fgd-embed3": [
        ("that_no-gap", "what_no-gap"),
        ("what_gap", "that_gap")
    ],
    "fgd-embed4": [
        ("that_no-gap", "what_no-gap"),
        ("what_gap", "that_gap")
    ],
    "fgd_hierarchy": [
        ("that_nogap", "what_nogap"),
        ("what_subjgap", "that_subjgap")
    ], # there's another formula that we're skipping
    "fgd_object": [
        ("that_nogap", "what_nogap"),
        ("what_gap", "that_gap")
    ],
    "fgd_pp": [
        ("that_nogap", "what_nogap"),
        ("what_gap", "that_gap")
    ],
    "fgd_subject": [
        ("that_nogap", "what_nogap"),
        ("what_gap", "that_gap")
    ],
    # skipping garden path
    "mvrr": [],
    "mvrr_mod": [],
    # skipping Roger's small test suite
    "nn-nv-rpl": [],
    # skipping NPI licensing -- prediction too complicated
    "npi_orc_any": [],
    "npi_orc_ever": [],
    "npi_src_any": [],
    "npi_src_ever": [],
    # skipping garden path
    "npz_ambig": [],
    "npz_ambig_mod": [],
    "npz_obj": [],
    "npz_obj_mod": [],
    "number_orc": [
        ("match_sing", "mismatch_sing"),
        ("match_plural", "mismatch_plural")
    ],
    "number_prep": [
        ("match_sing", "mismatch_sing"),
        ("match_plural", "mismatch_plural")
    ],
    "number_src": [
        ("match_sing", "mismatch_sing"),
        ("match_plural", "mismatch_plural")
    ],
    "reflexive_orc_fem": [
        ("match_sing", "mismatch_sing"),
        ("match_plural", "mismatch_plural")
    ],
    "reflexive_orc_masc": [
        ("match_sing", "mismatch_sing"),
        ("match_plural", "mismatch_plural")
    ],
    "reflexive_prep_fem": [
        ("match_sing", "mismatch_sing"),
        ("match_plural", "mismatch_plural")
    ],
    "reflexive_prep_masc": [
        ("match_sing", "mismatch_sing"),
        ("match_plural", "mismatch_plural")
    ],
    "reflexive_src_fem": [
        ("match_sing", "mismatch_sing"),
        ("match_plural", "mismatch_plural")
    ],
    "reflexive_src_masc": [
        ("match_sing", "mismatch_sing"),
        ("match_plural", "mismatch_plural")
    ],
    "subordination": [
        ("no-sub_no-matrix", "sub_no-matrix"),
        ("sub_matrix", "no-sub_matrix")
    ],
    "subordination_orc-orc": [
        ("no-sub_no-matrix", "sub_no-matrix"),
        ("sub_matrix", "no-sub_matrix")
    ],
    "subordination_pp-pp": [
        ("no-sub_no-matrix", "sub_no-matrix"),
        ("sub_matrix", "no-sub_matrix")
    ],
    "subordination_src-src": [
        ("no-sub_no-matrix", "sub_no-matrix"),
        ("sub_matrix", "no-sub_matrix")
    ],
}