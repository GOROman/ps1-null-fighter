cmake_minimum_required(VERSION 3.21)

project(
	nullfighter
	LANGUAGES    C ASM
	VERSION      0.1.0
	DESCRIPTION  "Skinned FBX character viewer for PlayStation 1"
)

# Convert both characters into the runtime formats.  Requires python3 with
# numpy + Pillow (+ xatlas for --reatlas) on the build host.
#   schoolgirl  un-rigged Tripo mesh: decimated to MODEL_TRIS triangles,
#               re-unwrapped UVs, auto rig + procedural animations
#   character   skinned + animated FBX
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(ASSET_SRC   ${PROJECT_SOURCE_DIR}/assets/src)
set(ASSET_OUT   ${PROJECT_BINARY_DIR}/assets)
set(MODEL_TRIS "1150" CACHE STRING "schoolgirl: target triangle count (0 = no decimation)")
set(TOOL_DEPS ${PROJECT_SOURCE_DIR}/tools/fbx2ps1.py ${PROJECT_SOURCE_DIR}/tools/fbxparse.py
              ${PROJECT_SOURCE_DIR}/tools/meshopt.py ${PROJECT_SOURCE_DIR}/tools/autorig.py
              ${PROJECT_SOURCE_DIR}/tools/sprite.py ${PROJECT_SOURCE_DIR}/tools/bodygen.py)

# convert_model(name fbx texture [FACE] extra-args...): FACE also writes
# <name>_face.tim (second texture page for the face)
function(convert_model name fbx tex)
	set(bin ${ASSET_OUT}/${name}.bin)
	set(tim ${ASSET_OUT}/${name}.tim)
	set(args ${ARGN})
	set(outputs ${bin} ${tim})
	set(face_args)
	list(FIND args FACE face_index)
	if(NOT face_index EQUAL -1)
		list(REMOVE_AT args ${face_index})
		set(face_args --face-tex --out-face-tim ${ASSET_OUT}/${name}_face.tim)
		list(APPEND outputs ${ASSET_OUT}/${name}_face.tim)
	endif()
	list(FIND args SKIRT skirt_index)
	if(NOT skirt_index EQUAL -1)
		list(REMOVE_AT args ${skirt_index})
		list(APPEND face_args --gen-skirt --out-skirt-tim ${ASSET_OUT}/${name}_skirt.tim)
		list(APPEND outputs ${ASSET_OUT}/${name}_skirt.tim)
	endif()
	add_custom_command(
		OUTPUT  ${outputs}
		COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSET_OUT}
		COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/fbx2ps1.py
		        ${fbx} --texture ${tex} ${args} ${face_args} --out-bin ${bin} --out-tim ${tim}
		DEPENDS ${TOOL_DEPS} ${fbx} ${tex} ${ASSET_SRC}/character.fbx
		COMMENT "Converting ${name} -> ${name}.bin / ${name}.tim"
		VERBATIM
	)
	set(MODEL_OUTPUTS ${MODEL_OUTPUTS} ${outputs} PARENT_SCOPE)
endfunction()

convert_model(schoolgirl ${ASSET_SRC}/schoolgirl/schoolgirl.fbx ${ASSET_SRC}/schoolgirl/basecolor.jpg
              FACE SKIRT --target-tris ${MODEL_TRIS} --reatlas --autorig
              --remove-hidden 0.02 --skirt 0.39 0.49 --gen-body)
convert_model(character ${ASSET_SRC}/character.fbx ${ASSET_SRC}/character.fbm/basecolor.jpg --front=+z
              --drop-bones thightwist,calftwist,upperarmtwist,forearmtwist,toebase,necktwist02)
# fight mode: a light version of the blonde (two of them must fit in a frame)
convert_model(character_lo ${ASSET_SRC}/character.fbx ${ASSET_SRC}/character.fbm/basecolor.jpg --front=+z
              --drop-bones thightwist,calftwist,upperarmtwist,forearmtwist,toebase,necktwist02
              --target-tris 600 --face-weight 100 --keep-eyes)
# third character: a Tripo rig without animations - the blonde's animations
# are transferred by bone name
convert_model(schoolgirl2 ${ASSET_SRC}/schoolgirl2/schoolgirl2.fbx ${ASSET_SRC}/schoolgirl2/basecolor.jpg
              --front=+z --anim-from ${ASSET_SRC}/character.fbx
              --drop-bones thightwist,calftwist,upperarmtwist,forearmtwist,toebase,necktwist02
              --target-tris 1400 --face-weight 100 --keep-eyes --tim-slot 1)
add_custom_command(
	OUTPUT  ${ASSET_OUT}/monkey.tim
	COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSET_OUT}
	COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tools/sprite.py ${ASSET_OUT}/monkey.tim
	DEPENDS ${PROJECT_SOURCE_DIR}/tools/sprite.py
	COMMENT "Generating monkey.tim"
	VERBATIM
)
add_custom_target(assets DEPENDS ${MODEL_OUTPUTS} ${ASSET_OUT}/monkey.tim)

psn00bsdk_add_executable(nullfighter GPREL
	src/main.c
	src/anim.c
	src/render.c
	src/camera.c
	src/ik.c
	src/walker.c
	src/fight.c
)
target_include_directories(nullfighter PRIVATE src)
target_link_libraries(nullfighter PRIVATE psxgpu psxgte psxetc psxapi)
add_dependencies(nullfighter assets)
psn00bsdk_target_incbin(nullfighter PRIVATE schoolgirl_bin ${ASSET_OUT}/schoolgirl.bin)
psn00bsdk_target_incbin(nullfighter PRIVATE schoolgirl_tim ${ASSET_OUT}/schoolgirl.tim)
psn00bsdk_target_incbin(nullfighter PRIVATE schoolgirl_face_tim ${ASSET_OUT}/schoolgirl_face.tim)
psn00bsdk_target_incbin(nullfighter PRIVATE schoolgirl_skirt_tim ${ASSET_OUT}/schoolgirl_skirt.tim)
psn00bsdk_target_incbin(nullfighter PRIVATE character_bin  ${ASSET_OUT}/character.bin)
psn00bsdk_target_incbin(nullfighter PRIVATE character_tim  ${ASSET_OUT}/character.tim)
psn00bsdk_target_incbin(nullfighter PRIVATE monkey_tim     ${ASSET_OUT}/monkey.tim)
psn00bsdk_target_incbin(nullfighter PRIVATE character_lo_bin ${ASSET_OUT}/character_lo.bin)
psn00bsdk_target_incbin(nullfighter PRIVATE schoolgirl2_bin ${ASSET_OUT}/schoolgirl2.bin)
psn00bsdk_target_incbin(nullfighter PRIVATE schoolgirl2_tim ${ASSET_OUT}/schoolgirl2.tim)

psn00bsdk_add_cd_image(
	iso
	nullfighter
	iso.xml
	DEPENDS nullfighter system.cnf
)
