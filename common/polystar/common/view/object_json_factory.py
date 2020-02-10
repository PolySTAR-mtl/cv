from polystar.common.models.object import Object, Armor, ObjectType, ArmorColor
from polystar.common.view.json_factory import JsonFactory, Json


class ObjectJsonFactory(JsonFactory[Object]):
    @staticmethod
    def from_json(json: Json) -> Object:
        t: ObjectType = ObjectType(json["name"])

        x, y, w, h = (
            int(float(json["bndbox"]["xmin"])),
            int(float(json["bndbox"]["ymin"])),
            int(float(json["bndbox"]["xmax"])) - int(float(json["bndbox"]["xmin"])),
            int(float(json["bndbox"]["ymax"])) - int(float(json["bndbox"]["ymin"])),
        )

        if t is not ObjectType.Armor:
            return Object(type=t, x=x, y=y, w=w, h=h)

        return Armor(type=t, x=x, y=y, w=w, h=h, numero=json["armor_class"], color=ArmorColor(json["armor_color"]))

    @staticmethod
    def to_json(obj: Object) -> Json:
        rv = {
            "name": obj.type.value.lower(),
            "bndbox": {"xmin": obj.x, "xmax": obj.x + obj.w, "ymin": obj.y, "ymax": obj.y + obj.h},
        }
        if isinstance(obj, Armor):
            rv.update({"armor_class": obj.numero, "armor_color": obj.color.value.lower()})
        return rv
